# src/sentence_chunking.py

import nltk
from transformers import AutoTokenizer
import re # Import regex for more advanced splitting if needed

# Import the fixed chunker function for handling oversized sentences
# Make sure this import path is correct for your project structure
from fixed_chunking import chunk_fixed_size

# --- Configuration ---
TOKENIZER_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"

# Download 'punkt' if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

# --- Initialize Tokenizer ---
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"DEBUG (sentence_chunking - Robust): Initialized tokenizer {TOKENIZER_NAME}")
except Exception as e:
    print(f"ERROR (sentence_chunking - Robust): Failed to initialize tokenizer {TOKENIZER_NAME}: {e}")
    # tokenizer remains None

# --- Helper Function for Consistent Token Counting ---
def _count_tokens_internal(text):
    """Helper to count tokens using the global tokenizer."""
    if tokenizer is None or not text:
        return 0
    try:
        # Consistent with main logic - includes special tokens for limit checks
        return len(tokenizer.encode(text, add_special_tokens=True))
    except Exception as e:
        print(f"Error during token counting: {e}")
        return 0

# --- New Helper Function for Hierarchical Splitting ---
def _split_long_sentence_hierarchically(sentence, max_tokens):
    """
    Tries to split a single sentence that exceeds max_tokens more intelligently.
    Attempts splits by strong punctuation first, then falls back to fixed chunking.
    """
    if tokenizer is None: # Need tokenizer for checking sub-segment lengths
         print("ERROR (sentence_chunking - Hierarchical Split): Tokenizer not available.")
         # Fallback to fixed chunker immediately if tokenizer failed globally
         return chunk_fixed_size(sentence, max_tokens)

    # 1. Define split delimiters in order of preference (strongest first)
    #    Regex patterns allow capturing the delimiter to potentially add it back
    #    (using non-capturing group `(?:...)` for lookbehind/lookahead if needed)
    #    Simpler approach first: Split by common clause separators.
    #    Using simple string split first for clarity. Regex can be added for more power.
    delimiters = [";", ":", "—", ". ", "? ", "! "] # Prioritize stronger breaks. Added space to sentence enders.

    segments = [sentence.strip()] # Start with the whole sentence as one segment

    for delim in delimiters:
        next_segments = []
        processed = False # Track if we actually split anything with this delimiter
        for segment in segments:
            # Only try splitting if the segment still exceeds the limit
            if _count_tokens_internal(segment) > max_tokens:
                # Simple split - loses the delimiter.
                # A regex approach like re.split(f'({re.escape(delim)})', segment)
                # could keep delimiters but complicates reassembly.
                parts = segment.split(delim)
                # Add non-empty parts back, effectively splitting by this delimiter
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part:
                         next_segments.append(part)
                         # Decide if adding delimiter back is needed - for now, keep it simple
                         # if i < len(parts) - 1: next_segments.append(delim.strip())
                processed = True # We performed splits
            else:
                # Segment fits, keep it as is for the next round
                next_segments.append(segment)

        segments = next_segments # Update segments for the next delimiter check
        # Optimization: If all segments now fit, no need to try weaker delimiters
        if processed and all(_count_tokens_internal(s) <= max_tokens for s in segments if s):
            break

    # 3. Final Check and Fallback
    final_chunks = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        segment_token_count = _count_tokens_internal(segment)
        if segment_token_count <= max_tokens:
            final_chunks.append(segment)
        else:
            # Fallback: Segment STILL too long after trying all semantic splits
            print(f"Warning (sentence_chunking): Segment still exceeds max_tokens ({segment_token_count} > {max_tokens}) after hierarchical split attempts. Using fixed chunker for: '{segment[:80]}...'")
            # Assuming chunk_fixed_size returns a list of strings
            sub_chunks = chunk_fixed_size(segment, max_tokens)
            final_chunks.extend(sub_chunks)

    return final_chunks

# --- Main Function (Modified) ---
def sentence_aware_chunking(text, max_tokens=512):
    """
    Splits text into chunks by grouping sentences using the configured tokenizer,
    respecting max_tokens (including special tokens). Sentences exceeding
    max_tokens are split hierarchically based on punctuation before falling
    back to fixed-size chunking.
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
        sentences = text.splitlines() # Basic fallback

    chunks = []
    current_chunk_sentences = [] # Accumulate sentences for the current chunk

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check token count for the sentence *itself* first
        sentence_token_count = _count_tokens_internal(sentence)

        if sentence_token_count > max_tokens:
            # --- Handle Sentence Exceeding Limit (MODIFIED BLOCK) ---
            print(f"Warning (sentence_chunking): Sentence exceeds max_tokens ({sentence_token_count} > {max_tokens}). Attempting hierarchical split.")
            # Finalize any pending chunk before processing the long sentence
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [] # Reset

            # Use the new hierarchical splitter for this oversized sentence
            # This function now handles the ultimate fallback to chunk_fixed_size internally
            sub_chunks = _split_long_sentence_hierarchically(sentence, max_tokens)
            chunks.extend(sub_chunks) # Add the resulting sub-chunks directly
            # --- END MODIFIED BLOCK ---
            continue # Move to the next sentence

        # --- Try adding the current sentence to the existing chunk ---
        # This logic remains the same as before
        potential_chunk_text = " ".join(current_chunk_sentences + [sentence])
        potential_token_count = _count_tokens_internal(potential_chunk_text)

        if potential_token_count <= max_tokens:
            # It fits, add sentence to the current list
            current_chunk_sentences.append(sentence)
        else:
            # It doesn't fit. Finalize the previous chunk.
            if current_chunk_sentences: # Ensure we don't add empty chunks
                chunks.append(" ".join(current_chunk_sentences))
            # Start new chunk list with the current sentence (which we know fits)
            current_chunk_sentences = [sentence]

    # Add the last remaining chunk if it's not empty
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks