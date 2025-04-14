# src/sentence_chunking.py

import nltk
from transformers import AutoTokenizer
# Import the fixed chunker function for handling oversized sentences
from fixed_chunking import chunk_fixed_size

# --- Configuration ---
TOKENIZER_NAME = "BAAI/bge-base-en-v1.5"

# Download 'punkt' if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

# --- Initialize Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"DEBUG (sentence_chunking - Robust): Initialized tokenizer {TOKENIZER_NAME}")
except Exception as e:
    print(f"ERROR (sentence_chunking - Robust): Failed to initialize tokenizer {TOKENIZER_NAME}: {e}")
    tokenizer = None

def sentence_aware_chunking(text, max_tokens=512):
    """
    Splits text into chunks by grouping sentences using the BGE tokenizer,
    respecting max_tokens (including special tokens) and handling sentences
    that individually exceed the limit by falling back to fixed-size chunking.
    """
    if tokenizer is None:
        print("ERROR (sentence_chunking - Robust): Tokenizer not available.")
        return []
    if not text:
        return []

    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        print(f"ERROR (sentence_chunking - Robust): NLTK sentence tokenization failed: {e}")
        # Fallback: split by newline, but this might still create long lines from tables
        sentences = text.splitlines()

    chunks = []
    current_chunk_sentences = [] # Use a list to accumulate sentences for the current chunk

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check token count for the sentence *itself* first (including special tokens)
        sentence_token_count = len(tokenizer.encode(sentence, add_special_tokens=True))

        if sentence_token_count > max_tokens:
            # --- Handle Sentence Exceeding Limit ---
            print(f"Warning (sentence_chunking): Sentence exceeds max_tokens ({sentence_token_count} > {max_tokens}). Splitting with fixed chunker.")
            # Finalize any pending chunk before processing the long sentence
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [] # Reset for safety

            # Use fixed chunking for this oversized sentence
            # chunk_fixed_size splits based on *content* tokens up to max_tokens
            sub_chunks = chunk_fixed_size(sentence, max_tokens) # Use max_tokens as the target size
            chunks.extend(sub_chunks) # Add the resulting sub-chunks directly
            continue # Move to the next sentence

        # --- Try adding the current sentence to the existing chunk ---
        potential_chunk_text = " ".join(current_chunk_sentences + [sentence])
        potential_token_count = len(tokenizer.encode(potential_chunk_text, add_special_tokens=True))

        if potential_token_count <= max_tokens:
            # It fits, add sentence to the current list
            current_chunk_sentences.append(sentence)
        else:
            # It doesn't fit. Finalize the previous chunk.
            if current_chunk_sentences: # Ensure we don't add empty chunks
                chunks.append(" ".join(current_chunk_sentences))
            # Start new chunk list with the current sentence (which we already know fits)
            current_chunk_sentences = [sentence]

    # Add the last remaining chunk if it's not empty
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks