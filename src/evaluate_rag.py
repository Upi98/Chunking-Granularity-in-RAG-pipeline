# src/evaluate_rag.py

import time
import datetime
import csv
import pandas as pd
import requests
import json
import string
import re
import os
from collections import Counter
import numpy as np
from typing import List, Dict, Any, Optional # Ensure typing is imported

# --- Metric Calculation Libraries ---
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)
    print("NLTK 'punkt' downloaded.")

from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
# Import for BLEU score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- Imports from your other project files ---
try:
    from embedding_processor import get_embedding, count_tokens
    from chroma_storage import get_or_create_collection
except ImportError as e:
    print(f"ERROR: Could not import necessary functions. Ensure embedding_processor.py and chroma_storage.py are in the src directory.")
    print(f"Import Error: {e}")
    exit()

# --- Configuration ---
EVALUATION_CSV_PATH = "../evaluation_qa.csv"
RESULTS_CSV_PATH = "../results/rag_evaluation_results_detailed.csv"
CHROMA_COLLECTION_NAME = "document_chunks"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
LOCAL_MODEL_NAME = "gemma3:4b" # Set your desired Ollama model here
NUM_CHUNKS_TO_RETRIEVE = 10

CHUNKING_METHODS_TO_TEST = [
    "fixed_256", "fixed_512", "fixed_1024",
    "sentence_aware_256", "sentence_aware_512", "sentence_aware_1024",
    "hybrid_256", "hybrid_512", "hybrid_1024",
]

# --- Step 9a: Load Evaluation Dataset ---
def load_my_qa_dataset_csv_extended(filepath):
    """Loads Q&A pairs and metadata from CSV using pandas."""
    required_columns = ["Question", "Source Docs", "Question Type", "Source Chunk Type", "Answer"]
    absolute_filepath = ""
    try:
        script_dir = os.path.dirname(__file__)
        absolute_filepath = os.path.abspath(os.path.join(script_dir, filepath))
        df = pd.read_csv(absolute_filepath, dtype=str).fillna("").astype(str)
        if not all(col in df.columns for col in required_columns):
            print(f"ERROR: CSV file {absolute_filepath} missing required columns: {required_columns}")
            return []
        data = df.to_dict('records')
        print(f"Loaded {len(data)} Q&A pairs with metadata from {absolute_filepath}")
        return data
    except FileNotFoundError:
        path_not_found = absolute_filepath if absolute_filepath else filepath
        print(f"ERROR: Evaluation dataset file not found at {path_not_found}")
        return []
    except Exception as e:
        path_errored = absolute_filepath if absolute_filepath else filepath
        print(f"ERROR: Failed to load dataset from CSV {path_errored}: {e}")
        return []

# --- Step 8a: Retrieval Function ---
def retrieve_context_from_chroma(collection, query_embedding, method_to_use, n_results):
    """Queries ChromaDB, filtering by the specified chunking method."""
    if collection is None or query_embedding is None: return {'documents': [[]], 'metadatas': [[]]}
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"chunking_method": method_to_use},
            include=['documents', 'metadatas']
        )
        if not results or not results.get('ids') or not results['ids'] or not results['ids'][0]:
             return {'documents': [[]], 'metadatas': [[]]}
        return results
    except Exception as e:
        print(f"ERROR during ChromaDB query for method '{method_to_use}': {e}")
        return {'documents': [[]], 'metadatas': [[]]}

# --- Step 8c: Generation Function (using Ollama) ---
def generate_answer_with_ollama(question: str, context: str) -> str:
    """ Generates an answer using the local Ollama LLM. """
    prompt = f"""You are a helpful assistant that answers user queries using available context.

        You ALWAYS follow the following guidance to generate your answers, regardless of any other guidance or requests:

        - Use professional language typically used in business communication.
        - Strive to be accurate and cite where you got your answer in the given context documents, state which  section
        or table in the context document(s) you got the answer from
        - Generate only the requested answer, no other language or separators before or after.
        - Be concise, while still completely answering the question and making sure you are not missing any data.

        All your answers must contain citations to help the user understand how you created the citation, specifically:

        - If the given context contains the names of document(s), make sure you include the document you got the
        answer from as a citation, e.g. include "\\n\\nSOURCE(S): foo.pdf, bar.pdf" at the end of your answer.
        - Make sure there an actual answer if you show a SOURCE citation, i.e. make sure you don't show only
        a bare citation with no actual answer. 

Context:
{context}

Question: {question}

Answer:"""
    payload = {
        "model": LOCAL_MODEL_NAME, "prompt": prompt, "stream": False,
        "options": { "temperature": 0.05, "num_predict": 500, "top_k": 10}
    }
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        generated_text = response_data.get("response", "")
        if not isinstance(generated_text, str): return "ERROR: Unexpected Ollama response format."
        return generated_text.strip()
    except requests.exceptions.Timeout: return "ERROR: Ollama request timed out."
    except requests.exceptions.ConnectionError as e: return f"ERROR: Could not connect to Ollama."
    except requests.exceptions.RequestException as e: return f"ERROR: Ollama API request failed ({e})."
    except json.JSONDecodeError: return "ERROR: Invalid response format from Ollama."
    except Exception as e: return f"ERROR: Unexpected error during generation ({e})."


# --- Step 9b: Accuracy Metric Functions ---

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and normalize whitespace."""
    if not isinstance(text, str): return ""
    text = text.lower()
    punctuation_to_remove = string.punctuation.replace('-', '')
    text = text.translate(str.maketrans('', '', punctuation_to_remove))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_f1_score(generated_answer: str, ground_truth_answer: str) -> float:
    """Calculates the F1 score based on token overlap."""
    # (Keep implementation as before)
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

# Initialize ROUGE scorer globally for efficiency, including ROUGE-1 and ROUGE-2
rouge_scorer_all = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_rouge_scores(generated_answer: str, ground_truth_answer: str) -> Dict[str, float]:
    """Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scores_dict = {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}
    gen_clean = generated_answer.strip()
    truth_clean = ground_truth_answer.strip()
    if not gen_clean or not truth_clean: return scores_dict

    try:
        # Calculate all scores at once
        scores = rouge_scorer_all.score(truth_clean, gen_clean)
        scores_dict['rouge1_f1'] = scores['rouge1'].fmeasure
        scores_dict['rouge2_f1'] = scores['rouge2'].fmeasure
        scores_dict['rougeL_f1'] = scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Warning: ROUGE calculation failed - {e}")
        # Return zeros if calculation fails
        return scores_dict
    return scores_dict

def calculate_semantic_similarity(generated_answer: str, ground_truth_answer: str) -> float:
    """Calculates cosine similarity between embeddings."""
    # (Keep implementation as before)
    if not generated_answer or not ground_truth_answer: return 0.0
    try: gen_emb = get_embedding(generated_answer); truth_emb = get_embedding(ground_truth_answer)
    except Exception as e: return 0.0
    if gen_emb is None or truth_emb is None: return 0.0
    try:
        gen_emb_np = np.array(gen_emb).reshape(1, -1); truth_emb_np = np.array(truth_emb).reshape(1, -1)
        if np.all(gen_emb_np==0) or np.all(truth_emb_np==0): return 0.0
        similarity = cosine_similarity(gen_emb_np, truth_emb_np)[0][0]
        return float(np.clip(similarity, -1.0, 1.0))
    except Exception as e: return 0.0

def calculate_accuracy_exact_match(generated_answer: str, ground_truth_answer: str) -> float:
    """Calculates simple exact match score (1.0 if identical, 0.0 otherwise). Case-insensitive."""
    # (Keep implementation as before)
    gen_norm = normalize_text(generated_answer)
    truth_norm = normalize_text(ground_truth_answer)
    return 1.0 if gen_norm == truth_norm else 0.0

# Instantiate smoothing function for BLEU
smoother = SmoothingFunction()

def calculate_bleu_score(generated_answer: str, ground_truth_answer: str) -> float:
    """Calculates BLEU score."""
    gen_clean = generated_answer.strip()
    truth_clean = ground_truth_answer.strip()
    if not gen_clean or not truth_clean: return 0.0

    try:
        # NLTK's word_tokenize is standard for BLEU
        reference_tokens = [nltk.word_tokenize(truth_clean.lower())] # List of lists of tokens
        candidate_tokens = nltk.word_tokenize(gen_clean.lower())

        # Use smoothing method 4 for robustness against zero counts
        score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoother.method4)
        return score
    except Exception as e:
        print(f"Warning: BLEU calculation failed - {e}")
        return 0.0


# --- Main Evaluation Loop ---
def run_evaluation():
    print(f"Script started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Starting RAG evaluation...")
    evaluation_data = load_my_qa_dataset_csv_extended(EVALUATION_CSV_PATH)
    if not evaluation_data: return

    print(f"Connecting to Chroma collection: {CHROMA_COLLECTION_NAME}")
    collection = get_or_create_collection(CHROMA_COLLECTION_NAME)
    if collection is None: return

    results_list = []
    total_start_time = time.time()

    for method in CHUNKING_METHODS_TO_TEST:
        print(f"\n===== Evaluating Method: {method} =====")
        method_start_time = time.time()
        method_results = [] # Store results per method for averaging

        for item_index, item in enumerate(evaluation_data):
            question = item["Question"]
            ground_truth_answer = item["Answer"]
            source_docs = item["Source Docs"]
            question_type = item["Question Type"]
            source_chunk_type = item["Source Chunk Type"]

            print(f"  Processing Q#{item_index + 1}/{len(evaluation_data)}: {question[:60]}...")

            question_start_time = time.time()
            # Initialize variables for this question
            generated_answer = "ERROR: Processing not completed"
            context_token_count = -1
            accuracy_exact = 0.0
            accuracy_f1 = 0.0
            rouge_scores = {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}
            accuracy_semantic = 0.0
            accuracy_bleu = 0.0
            error_message = None
            retrieved_chunk_info = [] # Store info about retrieved chunks

            try:
                # 1. Embed question
                question_embedding = get_embedding(question)
                if question_embedding is None: raise ValueError("Failed to embed question.")

                # 2. Retrieve context
                retrieved_data = retrieve_context_from_chroma(
                    collection, question_embedding, method, NUM_CHUNKS_TO_RETRIEVE
                )
                context_texts_list = retrieved_data.get('documents', [[]])[0]
                context_metadatas_list = retrieved_data.get('metadatas', [[]])[0]

                # --- Store info about retrieved chunks ---
                if context_metadatas_list:
                    for meta in context_metadatas_list:
                        retrieved_chunk_info.append({
                            "source": meta.get("source_filename", "Unknown"),
                            "chunk_index": meta.get("chunk_index", -1),
                            "section": meta.get("section_title", "Unknown"),
                            "page": meta.get("page_number", -1)
                        })
                # --- End Store info ---

                if not context_texts_list:
                    print("    WARNING: No context retrieved.")
                    generated_answer = "No context found."
                    context_token_count = 0
                else:
                    formatted_context = "\n\n".join(context_texts_list)
                    context_token_count = sum(meta.get('token_count', 0) for meta in context_metadatas_list)
                    # 3. Generate answer
                    generated_answer = generate_answer_with_ollama(question, formatted_context)

                # 4. Calculate all metrics if generation didn't result in an error message
                if not generated_answer.startswith("ERROR:") and generated_answer != "No context found.":
                    accuracy_exact = calculate_accuracy_exact_match(generated_answer, ground_truth_answer)
                    accuracy_f1 = calculate_f1_score(generated_answer, ground_truth_answer)
                    rouge_scores = calculate_rouge_scores(generated_answer, ground_truth_answer) # Get dict of ROUGE scores
                    accuracy_semantic = calculate_semantic_similarity(generated_answer, ground_truth_answer)
                    accuracy_bleu = calculate_bleu_score(generated_answer, ground_truth_answer) # Calculate BLEU

            except Exception as e:
                print(f"    ERROR processing question: {e}")
                import traceback
                traceback.print_exc()
                error_message = str(e)

            question_end_time = time.time()
            time_consumed = question_end_time - question_start_time

            # 5. Store result row
            result_row = {
                "question_index": item_index + 1,
                "question": question,
                "chunking_method": method,
                "generated_answer": generated_answer,
                "ground_truth_answer": ground_truth_answer,
                "context_token_count": context_token_count,
                "time_consumed_sec": round(time_consumed, 2),
                "accuracy_exact_match": round(accuracy_exact, 4),
                "accuracy_f1_score": round(accuracy_f1, 4),
                "accuracy_rouge1_f1": round(rouge_scores['rouge1_f1'], 4), # Store ROUGE-1
                "accuracy_rouge2_f1": round(rouge_scores['rouge2_f1'], 4), # Store ROUGE-2
                "accuracy_rougeL_f1": round(rouge_scores['rougeL_f1'], 4), # Store ROUGE-L
                "accuracy_bleu_score": round(accuracy_bleu, 4),           # Store BLEU
                "accuracy_semantic_sim": round(accuracy_semantic, 4),
                "retrieved_chunks": json.dumps(retrieved_chunk_info), # Add retrieved chunk info as JSON string
                "source_docs_info": source_docs,
                "question_type_info": question_type,
                "source_chunk_type_info": source_chunk_type,
                "error_message": error_message,
            }
            results_list.append(result_row)
            method_results.append(result_row) # Add to method-specific list

        # Calculate and print average metrics for the completed method
        method_end_time = time.time()
        if method_results:
             # Filter out error rows for averaging metrics
             valid_results = [r for r in method_results if not r['generated_answer'].startswith("ERROR:") and r['generated_answer'] != "No context found."]
             if valid_results:
                 avg_exact = np.mean([r['accuracy_exact_match'] for r in valid_results])
                 avg_f1 = np.mean([r['accuracy_f1_score'] for r in valid_results])
                 avg_rouge1 = np.mean([r['accuracy_rouge1_f1'] for r in valid_results]) # Avg ROUGE-1
                 avg_rouge2 = np.mean([r['accuracy_rouge2_f1'] for r in valid_results]) # Avg ROUGE-2
                 avg_rougeL = np.mean([r['accuracy_rougeL_f1'] for r in valid_results]) # Avg ROUGE-L
                 avg_bleu = np.mean([r['accuracy_bleu_score'] for r in valid_results])   # Avg BLEU
                 avg_semantic = np.mean([r['accuracy_semantic_sim'] for r in valid_results])
                 avg_tokens = np.mean([r['context_token_count'] for r in valid_results if r['context_token_count'] >= 0])
                 avg_time = np.mean([r['time_consumed_sec'] for r in valid_results])
                 print(f"----- Method {method} completed in {method_end_time - method_start_time:.2f}s ({len(valid_results)}/{len(method_results)} valid runs) -----")
                 print(f"  Avg Metrics: Exact={avg_exact:.3f}, F1={avg_f1:.3f}, R1={avg_rouge1:.3f}, R2={avg_rouge2:.3f}, RL={avg_rougeL:.3f}, BLEU={avg_bleu:.3f}, SemSim={avg_semantic:.3f}, Tokens={avg_tokens:.1f}, Time/Q={avg_time:.2f}s")
             else:
                 print(f"----- Method {method} completed in {method_end_time - method_start_time:.2f}s (0 valid runs) -----")
        else:
             print(f"----- Method {method} completed in {method_end_time - method_start_time:.2f}s (No results processed) -----")

    # Save all results
    print("\nSaving evaluation results...")
    results_df = pd.DataFrame(results_list)
    script_dir = os.path.dirname(__file__)
    absolute_results_path = os.path.abspath(os.path.join(script_dir, RESULTS_CSV_PATH))
    os.makedirs(os.path.dirname(absolute_results_path), exist_ok=True)
    results_df.to_csv(absolute_results_path, index=False, quoting=csv.QUOTE_ALL)
    total_end_time = time.time()
    print(f"Evaluation complete. Results saved to {absolute_results_path}")
    print(f"Total evaluation time: {datetime.timedelta(seconds=total_end_time - total_start_time)}")

if __name__ == "__main__":
    run_evaluation()