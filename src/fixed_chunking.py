# src/fixed_chunking.py
from transformers import AutoTokenizer

TOKENIZER_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
tokenizer = None # Initialize as None initially
try:
    # Attempt to load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"DEBUG (fixed_chunking): Initialized tokenizer {TOKENIZER_NAME}")
except Exception as e:
    print(f"ERROR (fixed_chunking): Failed to initialize tokenizer {TOKENIZER_NAME}: {e}")
    # tokenizer remains None if initialization fails

def chunk_fixed_size(text, chunk_size, overlap_ratio=0.1):
    # Check if initialization succeeded before using the tokenizer
    if tokenizer is None: # <--- This is where the error occurs
        print("ERROR (fixed_chunking): Tokenizer object is None. Cannot chunk.")
        return []
    if not text:
        return []
    # Encode WITHOUT special tokens to split based purely on content tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if not tokens:
        return []

    overlap = int(chunk_size * overlap_ratio)
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    chunks = []
    start = 0
    while start < len(tokens):
        # The chunk size refers to content tokens here
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        # Decode the content tokens. skip_special_tokens=True is correct here
        # as we didn't encode them in the first place for this splitting logic.
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        if end >= len(tokens):
            break # Reached the end

        next_start = start + chunk_size - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks

# Functions calling chunk_fixed_size remain the same
def chunk_fixed_256(text):
    return chunk_fixed_size(text, 256)

def chunk_fixed_512(text):
    return chunk_fixed_size(text, 512)

def chunk_fixed_1024(text):
    return chunk_fixed_size(text, 1024)
