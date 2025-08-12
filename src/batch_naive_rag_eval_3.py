# src/final_chunking_eval_with_all_metrics.py

import os
import re
import time
import datetime
import csv
import pandas as pd
import requests
import json
import string
import logging
import numpy as np
import tiktoken
from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError
from dotenv import load_dotenv
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import CrossEncoder

# --- NLTK and Metrics Imports ---
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    logging.info("NLTK 'punkt' not found. Downloading...")
    try: nltk.download('punkt', quiet=True); logging.info("'punkt' downloaded.")
    except Exception as e: logging.error(f"Failed to download nltk 'punkt': {e}")
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
# Import rouge_scorer
try:
    from rouge_score import rouge_scorer
except ImportError:
    logging.error("rouge-score library not found. Please install: pip install rouge-score")
    rouge_scorer = None # Set to None if import fails
from sklearn.metrics.pairwise import cosine_similarity

# --- Logging Setup ---
log_file = 'final_chunking_eval.log'
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING) # Add other noisy libraries if needed

# --- Configuration ---
load_dotenv(override=True)
NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI")
NEO4J_USER: Optional[str] = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
LOCAL_LLM_API_URL: Optional[str] = os.getenv("LOCAL_LLM_API_URL")
LOCAL_LLM_REQUEST_TIMEOUT: int = int(os.getenv("LOCAL_LLM_REQUEST_TIMEOUT", 240))

# --- Evaluation Parameters ---
LLM_MODEL = "mistral:7b-instruct-v0.3-q4_K_M"
CHUNKING_METHODS_TO_TEST: List[str] = [
    "fixed_256", "fixed_512", "fixed_1024",
    "sentence_aware_256", "sentence_aware_512", "sentence_aware_1024",
    "hybrid_modified_256", "hybrid_modified_512", "hybrid_modified_1024",
]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../evaluation_qa.csv"))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../results"))
OUTPUT_CSV_PATH = os.path.join(RESULTS_DIR, "final_chunking_naive_rag_all_metrics.csv") # Updated output name
TOP_K_CHUNKS = 5
INDEX_NAME = "chunk_embeddings"

# --- Tokenizer Setup ---
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    logging.info("Tiktoken tokenizer 'cl100k_base' loaded.")
except Exception as e:
    logging.warning(f"Failed to load tiktoken tokenizer: {e}. Using word count fallback.", exc_info=True); tokenizer = None
def count_tokens(text: str) -> int:
    if not isinstance(text, str): return 0
    if tokenizer and text:
        try: return len(tokenizer.encode(text))
        except Exception: return len(text.split())
    elif text: return len(text.split())
    else: return 0

try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logging.info("Cross-encoder reranker loaded.")
except Exception as e:
    logging.error(f"Failed to load cross-encoder reranker: {e}")
    reranker = None

# --- Embedding Function Import ---
try:
    from embedding_processor import get_embedding
    logging.info("Loaded get_embedding function successfully.")
except ImportError: logging.error("ERROR: embedding_processor.py not found."); exit(1)
except Exception as e: logging.error(f"Error importing get_embedding: {e}", exc_info=True); exit(1)

# --- Neo4j Driver Setup ---
driver: Optional[Driver] = None
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
     logging.critical("ERROR: Neo4j connection details missing."); exit(1)
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), max_connection_lifetime=7200, connection_timeout=60)
    driver.verify_connectivity(); logging.info(f"Neo4j connection successful to {NEO4J_URI}.")
except Exception as e: logging.critical(f"ERROR: Failed to create Neo4j driver: {e}", exc_info=True); driver = None; exit(1)

# --- LLM Interaction ---
def query_local_llm_with_metrics(model_name: str, prompt: str) -> Dict[str, Any]:
    """ Sends prompt to the specified local LLM, returns answer and metrics. """
    if not LOCAL_LLM_API_URL: return {"answer": None, "error": "LLM API URL not configured"}
    api_endpoint = LOCAL_LLM_API_URL
    is_chat_endpoint = "chat" in api_endpoint.split('/')[-1]
    options = { "temperature": 0.05, "num_predict": 600}
    payload = {"model": model_name, "stream": False, "options": options}
    if is_chat_endpoint: payload["messages"]= [{"role": "user", "content": prompt}]
    else: payload["prompt"]= prompt
    start_time = time.time()
    metrics = {"answer": None, "prompt_tokens": 0, "completion_tokens": 0, "llm_duration_ms": 0, "error": None}
    try:
        response = requests.post(api_endpoint, json=payload, timeout=LOCAL_LLM_REQUEST_TIMEOUT, headers={"Content-Type": "application/json"})
        response.raise_for_status(); response_data = response.json()
        answer = response_data.get("message", {}).get("content") if is_chat_endpoint else response_data.get("response")
        if answer and isinstance(answer, str): metrics["answer"] = answer.strip()
        else: metrics["error"] = "Empty/invalid response content"
        metrics["llm_prompt_tokens"] = response_data.get("prompt_eval_count", 0)
        metrics["llm_completion_tokens"] = response_data.get("eval_count", 0)
        metrics["llm_duration_ms"] = response_data.get("total_duration", 0) // 1_000_000
    except requests.exceptions.Timeout: metrics["error"] = f"Timeout ({LOCAL_LLM_REQUEST_TIMEOUT}s)"; logging.error(metrics["error"])
    except requests.exceptions.ConnectionError: metrics["error"] = "Connection error"; logging.error(f"{metrics['error']}. Is Ollama running?")
    except requests.exceptions.RequestException as e: metrics["error"] = f"Request error: {e}"; logging.error(metrics["error"])
    except Exception as e: metrics["error"] = f"Unexpected LLM error: {e}"; logging.error(metrics["error"], exc_info=True)
    if metrics["answer"] is None and metrics["error"] is None: metrics["error"] = "Unknown error"
    if metrics["llm_duration_ms"] == 0 and metrics["error"] is None : metrics["llm_duration_ms"] = round((time.time() - start_time) * 1000)
    return metrics


# --- Neo4j Retrieval ---
def run_query(tx: Transaction, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    logging.debug(f"Executing Cypher: {query[:100]}... with params: {parameters is not None}")
    result = tx.run(query, parameters or {})
    return [r.data() for r in result]

def top_k_chunks_filtered(tx: Transaction, question_embedding: List[float], k: int, chunking_method: str) -> List[Dict[str, Any]]:
    cypher_filter_after = """
    CALL db.index.vector.queryNodes($idx, $k_fetch, $vec)
    YIELD node AS chunk, score
    WHERE chunk.method = $chunking_method
    MATCH (doc:Document)-[:CONTAINS]->(chunk)
    RETURN chunk.text AS text, score, elementId(chunk) as node_id,
           doc.filename AS filename, chunk.is_table AS is_table
    ORDER BY score DESC LIMIT $k_limit
    """
    params_filter_after = {"idx": INDEX_NAME, "k_fetch": k * 10, "vec": question_embedding, "chunking_method": chunking_method, "k_limit": k}
    logging.debug(f"Running filtered vector query for method: {chunking_method}, K={k}")
    return run_query(tx, cypher_filter_after, params_filter_after)

def top_k_chunks_sparse(tx: Transaction, question: str, k: int, chunking_method: str) -> List[Dict[str, Any]]:
    cypher_sparse_query = """
    CALL db.index.fulltext.queryNodes('chunkTextFulltextIndex', $query)
    YIELD node AS chunk, score
    WHERE chunk.method = $chunking_method
    MATCH (doc:Document)-[:CONTAINS]->(chunk)
    RETURN chunk.text AS text, score, elementId(chunk) AS node_id,
           doc.filename AS filename, chunk.is_table AS is_table
    ORDER BY score DESC LIMIT $k
    """
    params = {"query": question, "chunking_method": chunking_method, "k": k}
    return run_query(tx, cypher_sparse_query, params)

def get_naive_context_with_metadata(tx: Transaction, question_embedding: List[float], k: int, chunking_method: str) -> Tuple[str, List[Dict[str, Any]]]:
    hits = top_k_chunks_filtered(tx, question_embedding, k, chunking_method)
    context_parts = [
        f"Source {i+1}:\n- Filename: {hit.get('filename', 'N/A')}\n- Is Table: {hit.get('is_table', False)}\n- Content: {hit.get('text', '')}"
        for i, hit in enumerate(hits)
    ]
    formatted_ctx = "\n\n---\n\n".join(context_parts)
    if not hits: logging.warning(f"No chunks found for method '{chunking_method}'.")
    return formatted_ctx, hits

# --- Generator Prompt ---
GENERATOR_PROMPT = """
You are a helpful assistant designed to answer questions based on provided context information (text passages or tables). Your primary goal is to derive answers from the context, but you can provide reasoned speculation when necessary.

**Processing Steps:**

1.  **Identify Key Entities:** First, carefully analyze the user's question to identify key entities such as specific company names, relevant time periods (e.g., year, quarter like Q1, Q2, Q3, Q4), financial metrics, products, or other specific subjects inquired about. Make note of these entities to guide your search.
2.  **Context-Based Search & Answering:** Using the identified key entities, search the provided context thoroughly for relevant information to construct the answer according to the rules below.

**Answering Rules:**

-   **Prioritize Context:** Always attempt to base your answer strictly on the information found within the provided context first.
-   **Professional Tone:** Use professional language suitable for business communication.
-   **Concise & Complete (as possible):** Be concise. Aim to completely address the question based on the context. If the context is insufficient, follow the rules below for partial answers or guesses.
-   **Synthesis (Within Context):** If multiple passages provide relevant information, synthesize them to form a more complete answer, citing all contributing sources. Do not make logical leaps or assumptions not explicitly supported by the text unless making a stated guess (see below).
-   **Partial Answers from Context:** If the context provides only a partial answer, present the available factual information clearly and explicitly state what specific information needed to fully answer the question is missing from the context.
-   **Insufficient Context & Educated Guesses:** If the context offers clues or related information but is insufficient for a factual answer, first state the relevant information found. Then, you may provide a *potential* answer based *only* on those context clues. **Crucially, you MUST explicitly label this as an "educated guess based on limited context" or "speculation based on available data."** Briefly explain the basis for the guess using only the context (e.g., "The context mentions X, which sometimes correlates with Y, so speculating Y might be..."). Do *not* introduce external knowledge or data for guesses.
-   **No Relevant Context:** If, after searching based on the identified key entities, the context contains *no relevant information* whatsoever to address the question, state: "Insufficient context provided to generate an answer." This phrasing is intended to deliberately reduce token overlap with typical reference answers.
-   **Citations:** Cite the specific source(s) (including filename or table indicator if provided) for *all* information drawn from the context, including facts used in partial answers and the contextual clues used as a basis for any educated guesses.

**Citation Requirements:**

-   Include a citation line at the end of your answer whenever context information is used:
    `\n\nSOURCE(S): [filename1], [filename2]`
-   Do not include any extra language, preamble, or separators—just the answer and the source citation if applicable.

Context:
---
{context}
---

Question: {question}

Answer:
"""

def get_hybrid_context(tx: Transaction, question: str, question_embedding: List[float], k: int, chunking_method: str) -> Tuple[str, List[Dict[str, Any]]]:
    dense_hits = top_k_chunks_filtered(tx, question_embedding, k, chunking_method)
    sparse_hits = top_k_chunks_sparse(tx, question, k, chunking_method)

    combined = {hit['node_id']: hit for hit in dense_hits}
    for hit in sparse_hits:
        if hit['node_id'] not in combined:
            combined[hit['node_id']] = hit

    sorted_hits = sorted(combined.values(), key=lambda x: x.get('score', 0), reverse=True)[:k]

    context_parts = [
        f"Source {i+1}:\n- Filename: {hit.get('filename', 'N/A')}\n- Is Table: {hit.get('is_table', False)}\n- Content: {hit.get('text', '')}"
        for i, hit in enumerate(sorted_hits)
    ]
    formatted_ctx = "\n\n---\n\n".join(context_parts)
    if not sorted_hits:
        logging.warning(f"[Hybrid Retrieval] No chunks found for method '{chunking_method}'.")
    return formatted_ctx, sorted_hits


# --- Metric Calculation Functions ---

# Normalize text helper
def normalize_text(text: str) -> str:
    # (Same as before)
    if not isinstance(text, str): return ""
    text = text.lower()
    punctuation_to_remove = string.punctuation.replace('-', '')
    text = text.translate(str.maketrans('', '', punctuation_to_remove))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# F1 Score
def calculate_f1_score(generated_answer: str, ground_truth_answer: str) -> float:
    # (Same as before)
    gen_norm = normalize_text(generated_answer)
    truth_norm = normalize_text(ground_truth_answer)
    if not gen_norm or not truth_norm: return 0.0
    gen_tokens = Counter(gen_norm.split())
    truth_tokens = Counter(truth_norm.split())
    if not gen_tokens or not truth_tokens: return 0.0
    common_tokens = gen_tokens & truth_tokens
    num_common = sum(common_tokens.values())
    if num_common == 0: return 0.0
    precision = num_common / sum(gen_tokens.values()) if sum(gen_tokens.values()) > 0 else 0.0
    recall = num_common / sum(truth_tokens.values()) if sum(truth_tokens.values()) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

# ROUGE Scorer Initialization (Modified)
try:
    # Initialize for ROUGE-1, ROUGE-2, and ROUGE-L
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    logging.info("Initialized rouge_scorer for ROUGE-1, ROUGE-2, ROUGE-L.")
except Exception as e:
     logging.error(f"Failed to initialize ROUGE scorer: {e}. ROUGE scores will be 0.", exc_info=True)
     rouge_scorer_instance = None

# ROUGE Calculation Function (Modified)
def calculate_rouge_scores(generated_answer: str, ground_truth_answer: str) -> Dict[str, float]:
    """Calculates ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scores = {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0} # Default scores
    if not rouge_scorer_instance: return scores

    gen_clean = generated_answer.strip() if isinstance(generated_answer, str) else ""
    truth_clean = ground_truth_answer.strip() if isinstance(ground_truth_answer, str) else ""
    if not gen_clean or not truth_clean: return scores

    try:
        rouge_results = rouge_scorer_instance.score(truth_clean, gen_clean)
        scores['rouge1_f1'] = rouge_results['rouge1'].fmeasure
        scores['rouge2_f1'] = rouge_results['rouge2'].fmeasure
        scores['rougeL_f1'] = rouge_results['rougeL'].fmeasure
    except Exception as e:
        logging.warning(f"ROUGE calculation failed: {e}")
        # Keep default scores (0.0)

    return scores

# Semantic Similarity
def calculate_semantic_similarity(generated_answer: str, ground_truth_answer: str) -> float:
    gen_clean = generated_answer.strip() if isinstance(generated_answer, str) else ""
    truth_clean = ground_truth_answer.strip() if isinstance(ground_truth_answer, str) else ""
    if not gen_clean or not truth_clean: return 0.0
    try:
        gen_emb = get_embedding(gen_clean); truth_emb = get_embedding(truth_clean)
        if gen_emb is None or truth_emb is None: logging.warning("Embedding failed for semantic sim."); return 0.0
        gen_emb_np = np.array(gen_emb).reshape(1, -1); truth_emb_np = np.array(truth_emb).reshape(1, -1)
        if np.all(gen_emb_np==0) or np.all(truth_emb_np==0): logging.warning("Zero vector in semantic sim."); return 0.0
        similarity = cosine_similarity(gen_emb_np, truth_emb_np)[0][0]
        return float(np.clip(similarity, -1.0, 1.0))
    except ValueError as e: logging.warning(f"Semantic sim value error: {e}"); return 0.0
    except Exception as e: logging.warning(f"Cosine similarity calc failed: {e}", exc_info=True); return 0.0

# Exact Match
def calculate_accuracy_exact_match(generated_answer: str, ground_truth_answer: str) -> float:
    gen_norm = normalize_text(generated_answer)
    truth_norm = normalize_text(ground_truth_answer)
    return 1.0 if gen_norm == truth_norm and gen_norm != "" else 0.0

# BLEU Score
def calculate_bleu_score(generated_answer: str, ground_truth_answer: str) -> float:
    gen_clean = generated_answer.strip() if isinstance(generated_answer, str) else ""
    truth_clean = ground_truth_answer.strip() if isinstance(ground_truth_answer, str) else ""
    if not gen_clean or not truth_clean: return 0.0
    try:
        reference_tokens = [word_tokenize(truth_clean)]
        candidate_tokens = word_tokenize(gen_clean)
        # Handle empty lists after tokenization
        if not candidate_tokens: return 0.0
        smoother = SmoothingFunction().method4 # method4 is robust
        score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoother)
        return score
    except Exception as e: logging.warning(f"BLEU score calculation failed: {e}"); return 0.0


# --- Main RAG Function (Calls all metric functions) ---
def run_single_rag_evaluation(session: Session, question: str, reference_answer: str, chunking_method: str, k: int) -> Dict[str, Any]:
    """ Runs the naive RAG pipeline, calculates all metrics including BLEU, ROUGE-1/2/L, and includes re-ranking. """
    overall_qa_start_time = time.time()
    metrics = {
        "generated_answer": None, "retrieval_time_s": 0.0, "query_embedding_input_tokens": 0,
        "retrieved_context_tokens": 0, "llm_prompt_tokens": 0, "llm_completion_tokens": 0,
        "llm_duration_ms": 0, "total_qa_time_s": 0.0, "error": None, "retrieved_chunk_count": 0,
        "accuracy_exact_match": 0.0, "accuracy_f1_score": 0.0,
        "accuracy_rouge1_f1": 0.0, "accuracy_rouge2_f1": 0.0, "accuracy_rougeL_f1": 0.0,
        "accuracy_semantic_sim": 0.0, "accuracy_bleu": 0.0,
        "retrieved_filenames": [], "retrieved_is_table": []
    }
    context_text = ""
    retrieved_hits = []

    try:
        # 1. Embed Query & Count Tokens
        query_for_embedding = question
        metrics["query_embedding_input_tokens"] = count_tokens(query_for_embedding)
        question_embedding = get_embedding(query_for_embedding)
        if question_embedding is None:
            raise ValueError("Failed to get question embedding.")

        # 2. Retrieve Context and Apply Cross-Encoder Re-Ranking
        retrieval_start_time = time.time()
        context_text, retrieved_hits = session.execute_read(get_hybrid_context, question, question_embedding, k * 3, chunking_method)
        metrics["retrieval_time_s"] = round(time.time() - retrieval_start_time, 4)

        # Re-rank using CrossEncoder
        if reranker and retrieved_hits:
            query_chunk_pairs = [(question, h["text"]) for h in retrieved_hits]
            relevance_scores = reranker.predict(query_chunk_pairs)
            for i, h in enumerate(retrieved_hits):
                h["rerank_score"] = float(relevance_scores[i])
            retrieved_hits = sorted(retrieved_hits, key=lambda x: x["rerank_score"], reverse=True)[:k]

        # Finalize context
        context_parts = [
            f"Source {i+1}:\n- Filename: {hit.get('filename', 'N/A')}\n- Is Table: {hit.get('is_table', False)}\n- Content: {hit.get('text', '')}"
            for i, hit in enumerate(retrieved_hits)
        ]
        context_text = "\n\n---\n\n".join(context_parts)
        metrics["retrieved_context_tokens"] = count_tokens(context_text)
        metrics["retrieved_chunk_count"] = len(retrieved_hits)
        metrics["retrieved_filenames"] = [h.get("filename", "N/A") for h in retrieved_hits]
        metrics["retrieved_is_table"] = [h.get("is_table", None) for h in retrieved_hits]

        # 3. Call LLM Generator
        if context_text:
            prompt = GENERATOR_PROMPT.format(context=context_text, question=question)
            llm_result = query_local_llm_with_metrics(LLM_MODEL, prompt)
            if "answer" in llm_result:
                llm_result["generated_answer"] = llm_result.pop("answer")
            metrics.update(llm_result)
        else:
            metrics["error"] = "No context retrieved"
            metrics["generated_answer"] = "No context found."

    except Exception as e:
        logging.error(f"Error during RAG process for method '{chunking_method}', Q: '{question[:50]}...': {e}", exc_info=True)
        if not metrics.get("error"):
            metrics["error"] = f"RAG pipeline error: {e}"

    # 4. Calculate Accuracy Metrics
    generated_answer = metrics.get("generated_answer")
    if generated_answer and not metrics.get("error"):
        try:
            metrics["accuracy_exact_match"] = calculate_accuracy_exact_match(generated_answer, reference_answer)
            metrics["accuracy_f1_score"] = calculate_f1_score(generated_answer, reference_answer)
            rouge_scores = calculate_rouge_scores(generated_answer, reference_answer)
            metrics["accuracy_rouge1_f1"] = rouge_scores['rouge1_f1']
            metrics["accuracy_rouge2_f1"] = rouge_scores['rouge2_f1']
            metrics["accuracy_rougeL_f1"] = rouge_scores['rougeL_f1']
            metrics["accuracy_semantic_sim"] = calculate_semantic_similarity(generated_answer, reference_answer)
            metrics["accuracy_bleu"] = calculate_bleu_score(generated_answer, reference_answer)
        except Exception as e:
            logging.error(f"Error calculating metrics for Q, Method '{chunking_method}': {e}", exc_info=True)
            if not metrics.get("error"):
                metrics["error"] = f"Metric calculation error: {e}"

    metrics["total_qa_time_s"] = round(time.time() - overall_qa_start_time, 4)
    return metrics


# --- Main Evaluation Loop ---
def run_chunking_evaluation():
    # (Setup checks as before)
    if not driver: logging.critical("Neo4j driver not available."); return
    if not os.path.exists(INPUT_CSV_PATH): logging.critical(f"Input file not found: {INPUT_CSV_PATH}."); return
    os.makedirs(RESULTS_DIR, exist_ok=True); logging.info(f"Results dir: {RESULTS_DIR}")
    # --- Load CSV ---
    # (Load CSV as before)
    logging.info(f"Loading evaluation questions from: {INPUT_CSV_PATH}")
    try:
        eval_df = pd.read_csv(INPUT_CSV_PATH, dtype=str).fillna("")
        required_cols = ["Question", "Answer", "Source Docs", "Question Type", "Source Chunk Type"]
        if not all(col in eval_df.columns for col in required_cols): raise ValueError(f"CSV missing columns: {required_cols}")
        logging.info(f"Loaded {len(eval_df)} questions.")
    except Exception as e: logging.critical(f"Failed to load/validate CSV: {e}", exc_info=True); return

    all_results = []
    total_start_time = time.time()
    logging.info(f"--- Starting Evaluation ---")
    logging.info(f"LLM Model: {LLM_MODEL}")
    logging.info(f"Chunking Methods: {', '.join(CHUNKING_METHODS_TO_TEST)}")

    # --- Loop through Methods ---
    for method_index, method_name in enumerate(CHUNKING_METHODS_TO_TEST):
        logging.info(f"===== Starting Method {method_index+1}/{len(CHUNKING_METHODS_TO_TEST)}: {method_name} =====")
        method_start_time = time.time()
        method_results_list = []

        # --- Loop through Questions ---
        for index, row in eval_df.iterrows():
            question = row['Question']
            reference_answer = row['Answer']

            logging.info(f"  Processing Q{index+1}/{len(eval_df)} for method '{method_name}'...")

            rag_metrics = {}
            try:
                with driver.session(database=NEO4J_DATABASE) as session:
                     rag_metrics = run_single_rag_evaluation(session, question, reference_answer, method_name, TOP_K_CHUNKS)
            except Exception as e:
                 logging.error(f"Critical error running RAG for Q{index+1}, Method '{method_name}': {e}", exc_info=True)
                 rag_metrics = {"error": f"Outer execution error: {e}"}

            # Store result row - Added ROUGE-1, ROUGE-2 scores
            result_row = {
                **row.to_dict(), # Original CSV columns
                "Chunking Method": method_name,
                "Model": LLM_MODEL,
                "Retrieval Method": "Hybrid (Dense + Sparse)",
                "Generated Answer": rag_metrics.get("generated_answer"),
                "Retrieval Time (s)": rag_metrics.get("retrieval_time_s"),
                "Query Embedding Input Tokens": rag_metrics.get("query_embedding_input_tokens"),
                "Retrieved Context Tokens": rag_metrics.get("retrieved_context_tokens"),
                "LLM Prompt Tokens": rag_metrics.get("llm_prompt_tokens"),
                "LLM Completion Tokens": rag_metrics.get("llm_completion_tokens"),
                "LLM Duration (ms)": rag_metrics.get("llm_duration_ms"),
                "Total QA Time (s)": rag_metrics.get("total_qa_time_s"),
                "Retrieved Chunk Count": rag_metrics.get("retrieved_chunk_count"),
                "Retrieved Filenames": json.dumps(rag_metrics.get("retrieved_filenames", [])),
                "Retrieved Is_Table Flags": json.dumps(rag_metrics.get("retrieved_is_table", [])),
                "Accuracy Exact Match": rag_metrics.get("accuracy_exact_match"),
                "Accuracy F1 Score": rag_metrics.get("accuracy_f1_score"),
                "Accuracy ROUGE-1 F1": rag_metrics.get("accuracy_rouge1_f1"), # Added
                "Accuracy ROUGE-2 F1": rag_metrics.get("accuracy_rouge2_f1"), # Added
                "Accuracy ROUGE-L F1": rag_metrics.get("accuracy_rougeL_f1"), # Kept
                "Accuracy Semantic Sim": rag_metrics.get("accuracy_semantic_sim"),
                "Accuracy BLEU": rag_metrics.get("accuracy_bleu"),
                "Error": rag_metrics.get("error")
            }
            all_results.append(result_row)
            method_results_list.append(result_row) # For averaging

            if (index + 1) % 10 == 0: logging.info(f"    ... completed {index+1} questions for {method_name}.")

        # --- Calculate and Print Average Metrics (Added ROUGE-1, ROUGE-2) ---
        method_duration = time.time() - method_start_time
        if method_results_list:
             try:
                temp_df = pd.DataFrame(method_results_list)
                avg_exact = temp_df.get('Accuracy Exact Match', pd.Series(dtype='float64')).mean()
                avg_f1 = temp_df.get('Accuracy F1 Score', pd.Series(dtype='float64')).mean()
                avg_rouge1 = temp_df.get('Accuracy ROUGE-1 F1', pd.Series(dtype='float64')).mean() # Added
                avg_rouge2 = temp_df.get('Accuracy ROUGE-2 F1', pd.Series(dtype='float64')).mean() # Added
                avg_rougeL = temp_df.get('Accuracy ROUGE-L F1', pd.Series(dtype='float64')).mean()
                avg_semantic = temp_df.get('Accuracy Semantic Sim', pd.Series(dtype='float64')).mean()
                avg_bleu = temp_df.get('Accuracy BLEU', pd.Series(dtype='float64')).mean()
                valid_context_tokens = temp_df[temp_df['Retrieved Context Tokens'] >= 0]['Retrieved Context Tokens']
                avg_context_tokens = valid_context_tokens.mean() if not valid_context_tokens.empty else 0
                avg_time = temp_df['Total QA Time (s)'].mean()
                error_count = temp_df['Error'].notna().sum()

                logging.info(f"----- Method {method_name} completed in {method_duration:.2f}s -----")
                logging.info(f"  Avg Metrics: BLEU={avg_bleu:.3f}, R1={avg_rouge1:.3f}, R2={avg_rouge2:.3f}, RL={avg_rougeL:.3f}, F1={avg_f1:.3f}, SemSim={avg_semantic:.3f}, Exact={avg_exact:.3f}") # Added R1, R2
                logging.info(f"  Avg Tokens: Context={avg_context_tokens:.1f}")
                logging.info(f"  Avg Time/Q={avg_time:.2f}s | Errors: {error_count}/{len(method_results_list)}")
             except Exception as e:
                  logging.error(f"Could not calculate average metrics for method {method_name}: {e}")
        else:
             logging.info(f"----- Method {method_name} completed in {method_duration:.2f}s (No results processed) -----")


        # --- Save intermediate results ---
        intermediate_output_path = os.path.join(RESULTS_DIR, f"results_method_{method_name}_all_metrics.csv") # Updated name
        logging.info(f"Saving intermediate results for method '{method_name}' to {intermediate_output_path}...")
        try:
            method_results_df = pd.DataFrame(method_results_list)
            method_results_df.to_csv(intermediate_output_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
            logging.info(f"Intermediate results for {method_name} saved.")
        except Exception as e:
            logging.error(f"Failed to save intermediate results for {method_name}: {e}", exc_info=True)

    # --- Save final results ---
    final_output_path = os.path.join(RESULTS_DIR, "final_chunking_naive_rag_all_metrics.csv") # Use consistent name
    logging.info(f"Saving {len(all_results)} total results to {final_output_path}...")
    try:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(final_output_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        logging.info("Final results saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save final results to CSV: {e}", exc_info=True)

    total_duration = time.time() - total_start_time
    logging.info(f"--- Full chunking strategy evaluation complete in {total_duration:.2f} seconds ({datetime.timedelta(seconds=total_duration)}) ---")


# --- Main Execution ---
if __name__ == "__main__":
    run_chunking_evaluation()
    if driver:
        driver.close()
        logging.info("Neo4j connection closed.")