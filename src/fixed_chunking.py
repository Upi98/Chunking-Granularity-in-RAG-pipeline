# src/fixed_chunking.py
import tiktoken

def chunk_fixed_size(text, chunk_size, overlap_ratio=0.1):
    """
    Splits text into fixed-size chunks with an overlap.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    overlap = int(chunk_size * overlap_ratio)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        # Move start pointer back by the overlap amount.
        start = end - overlap
    return chunks

def chunk_fixed_256(text):
    return chunk_fixed_size(text, 256)

def chunk_fixed_512(text):
    return chunk_fixed_size(text, 512)

def chunk_fixed_1024(text):
    return chunk_fixed_size(text, 1024)
