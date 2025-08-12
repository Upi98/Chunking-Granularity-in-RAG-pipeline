# src/embedding_processor.py

import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from typing import List, Optional
import numpy as np

# --- Model and Tokenizer Configuration ---
LOCAL_EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
EMBEDDING_DIMENSION = 768
DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
# Use the same identifier for the tokenizer
TOKENIZER_NAME = LOCAL_EMBEDDING_MODEL

# --- Initialize Model and Tokenizer ---
model = None
tokenizer = None # Initialize tokenizer variable
try:
    print(f"DEBUG: Initializing SentenceTransformer model '{LOCAL_EMBEDDING_MODEL}' on device '{DEVICE}'...")
    model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device=DEVICE)
    print(f"DEBUG: SentenceTransformer model initialized successfully.")
    # Initialize Tokenizer
    print(f"DEBUG: Initializing Hugging Face tokenizer '{TOKENIZER_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"DEBUG: Hugging Face tokenizer initialized successfully.")

    # Optional dimension check (keep as is)
    test_emb = model.encode("test")
    if len(test_emb) != EMBEDDING_DIMENSION:
         print(f"Warning: Expected dimension {EMBEDDING_DIMENSION} but got {len(test_emb)}")
    else:
        print(f"DEBUG: Model embedding dimension confirmed: {len(test_emb)}")

except Exception as e:
    print(f"ERROR: Failed to initialize model or tokenizer: {e}")
    print("Ensure 'sentence-transformers', 'transformers', and 'torch' are installed.")
    model = None
    tokenizer = None # Ensure tokenizer is None if error occurs

# --- Embedding Function --- (Keep as is)
def get_embedding(text: str) -> Optional[List[float]]:
    # ... (function code remains the same) ...
    if model is None: return None # Check added during init
    if not text or not text.strip(): return [0.0] * EMBEDDING_DIMENSION # Check added during init
    try:
        processed_text = ' '.join(text.split())
        embedding_np = model.encode(processed_text, normalize_embeddings=True)
        embedding = embedding_np.tolist()
        return embedding
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during local embedding generation: {e}")
        return None

# --- Token Counting Function (MODIFIED) ---
def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a given text using the loaded Hugging Face tokenizer.
    """
    if tokenizer is None:
        print("ERROR: Tokenizer not initialized. Cannot count tokens.")
        # Fallback or raise error - returning len(text)//4 might be misleading now
        return 0 # Or raise an exception

    if not text:
        return 0
    try:
        # Use the Hugging Face tokenizer's encode method (returns list of token IDs)
        # add_special_tokens=False might be desired if counting purely for chunking limits,
        # but True is often default and might be what chunk limits implicitly expect. Test needed.
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        print(f"Error during token counting with HF tokenizer: {e}")
        return 0 # Or raise error
