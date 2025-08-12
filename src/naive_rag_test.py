# src/naive_rag_test.py
import os
import textwrap
import requests # Added
import json     # Added
import logging  # Added
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Configuration ---
load_dotenv(override=True)
NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI")
NEO4J_USER: Optional[str] = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")

# --- LLM Configuration (Needs to be in your .env file) ---
LOCAL_LLM_API_URL: Optional[str] = os.getenv("LOCAL_LLM_API_URL", "http://localhost:11434/api/generate") # Default added
LOCAL_LLM_MODEL: Optional[str] = os.getenv("LOCAL_LLM_MODEL", "mistral:7b-instruct-v0.3-q4_K_M")      # Default added
LOCAL_LLM_REQUEST_TIMEOUT: int = int(os.getenv("LOCAL_LLM_REQUEST_TIMEOUT", 180))

# --- Embedding Setup ---
INDEX_NAME = "chunk_embeddings" # Make sure this vector index exists
try:
    # Adjust the import path based on your project structure
    from embedding_processor import get_embedding
except ImportError:
    logging.error("ERROR: Ensure embedding_processor.py and its get_embedding function are accessible.")
    def get_embedding(text: str) -> List[float]: # Dummy fallback
        logging.warning("WARN: Using dummy get_embedding function!")
        return [0.0] * 768 # Replace 768 with your actual embedding dimension

# --- Neo4j Driver Setup ---
driver: Optional[Driver] = None
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
     logging.error("ERROR: Neo4j connection details missing in environment variables.")
else:
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
        logging.info("Neo4j connection successful.")
    except Exception as e:
        logging.error(f"ERROR: Failed to create Neo4j driver: {e}")
        driver = None

# --- LLM API Function ---
def query_local_llm(prompt: str) -> Optional[str]:
    """ Sends a prompt to the local LLM and returns the text response. """
    if not LOCAL_LLM_API_URL or not LOCAL_LLM_MODEL:
        logging.error("Local LLM API URL or Model Name not configured.")
        return None

    # Determine payload structure (simplified for generator)
    api_endpoint = LOCAL_LLM_API_URL
    is_chat_endpoint = "chat" in api_endpoint.split('/')[-1]

    if is_chat_endpoint:
        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # No "format": "json" needed here - we want the text answer
            # "options": { "temperature": 0.3 } # Optional generation params
        }
    else: # Assume generate endpoint
        payload = {
            "model": LOCAL_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            # "options": { "temperature": 0.3 }
        }

    llm_output_str = None
    try:
        response = requests.post(
            api_endpoint,
            json=payload,
            timeout=LOCAL_LLM_REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        response_data = response.json()

        if is_chat_endpoint:
             message = response_data.get("message", {})
             llm_output_str = message.get("content") if isinstance(message, dict) else None
        else: # Assume generate endpoint
             llm_output_str = response_data.get("response")

        if not llm_output_str or not isinstance(llm_output_str, str) or llm_output_str.isspace():
             logging.warning(f"LLM response content is empty or not a string. Raw response: {response_data}")
             return None

        return llm_output_str.strip()

    except requests.exceptions.Timeout:
        logging.error(f"Timeout error querying local LLM API ({api_endpoint}) after {LOCAL_LLM_REQUEST_TIMEOUT}s.")
        return None
    except requests.exceptions.ConnectionError:
         logging.error(f"Connection error querying local LLM API ({api_endpoint}). Is Ollama running?")
         return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying local LLM API ({api_endpoint}): {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during LLM query: {e}", exc_info=True)
        return None


# --- Neo4j Retrieval Functions ---
def top_k_chunks(question: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return top‑k {text, score} dicts (vector search only)."""
    if not driver:
        logging.error("Neo4j driver not available for top_k_chunks.")
        return []

    # Adjust prefix if needed for your specific embedding model/function
    embedding_prefix = "Represent this sentence for searching relevant passages: "
    try:
        q_emb = get_embedding(embedding_prefix + question)
    except Exception as e:
        logging.error(f"Failed to get embedding for question: {e}")
        return []

    # Removed the embedding_model filter assuming it doesn't exist on nodes
    cypher = """
    CALL db.index.vector.queryNodes($idx, $k, $vec)
    YIELD node, score
    RETURN node.text AS text, score
    ORDER BY score DESC
    """
    params = {
        "idx": INDEX_NAME,
        "k": k,
        "vec": q_emb
    }
    try:
        with driver.session() as s:
            result = s.run(cypher, params)
            return result.data() # List of dictionaries
    except Exception as e:
        logging.error(f"Failed to run Neo4j vector query: {e}")
        return []

def naive_context(question: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    """Gets top-k chunks and joins their text for context."""
    hits = top_k_chunks(question, k)
    # Handle cases where hits might be empty or text is missing
    ctx = "\n\n---\n\n".join(h.get("text", "") for h in hits if h.get("text"))
    return ctx, hits # second item is metadata


# --- Generator Prompt ---
GENERATOR_PROMPT = """
You are a helpful assistant. Answer the following question based *only* on the provided context information. If the context does not contain the answer, say "I cannot answer the question based on the provided context." Do not use any prior knowledge.

Context:
---
{context}
---

Question: {question}

Answer:
"""

# --- Main RAG Function ---
def generate_naive_rag_answer(question: str, k: int = 5) -> Optional[str]:
    """Performs naive RAG: retrieves k chunks, formats prompt, calls LLM."""
    logging.info(f"Performing naive RAG for question: '{question}'")

    # 1. Retrieve Naive Context
    context_text, _ = naive_context(question, k) # We only need the text string here

    if not context_text:
        logging.warning("No context retrieved from vector search.")
        # Optionally return a specific message or None
        return "Could not retrieve relevant context from the database."

    # 2. Format Prompt
    prompt = GENERATOR_PROMPT.format(context=context_text, question=question)

    # 3. Call LLM Generator
    logging.info(f"Sending request to LLM: {LOCAL_LLM_MODEL}...")
    answer = query_local_llm(prompt)

    return answer


# --- Main Execution Block ---
if __name__ == "__main__":
    if not driver:
        logging.critical("Exiting: Neo4j driver not initialized.")
    elif not LOCAL_LLM_API_URL or not LOCAL_LLM_MODEL:
         logging.critical("Exiting: Local LLM not configured in .env file.")
    else:
        test_question = "In the Q2 2023 10-Q, did Microsoft announce any major business acquisitions or spin-offs??"
        # Or try another question relevant to your data, e.g., related to Intel
        # test_question = "What were Intel's key challenges in Q2 2023?"

        print(f"\n--- Running Naive RAG Test ---")
        print(f"Question: {test_question}")

        generated_answer = generate_naive_rag_answer(test_question, k=3) # Use k=3 for concise context

        print("\n--- Generated Answer ---")
        if generated_answer:
            # Use textwrap for better readability
            wrapper = textwrap.TextWrapper(width=100)
            print(wrapper.fill(generated_answer))
        else:
            print("Failed to generate an answer from the LLM.")

        # Close Neo4j driver
        driver.close()
        print("\n--- Test Complete ---")