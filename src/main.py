# src/main.py

import os
import glob
# No longer need tiktoken here if store_chunks was updated
# import tiktoken
from fixed_chunking import chunk_fixed_256, chunk_fixed_512, chunk_fixed_1024
# Import the parameterized functions
from sentence_chunking import sentence_aware_chunking
from hybrid_chunking import hybrid_element_semantic_chunking
# Imports for embedding and storage
from embedding_processor import get_embedding
from neo4j_storage import store_chunks
from pdf_extractor import extract_text_from_pdf


def load_raw_text(file_path):
    """
    Loads raw text from a file. Uses PDF extraction if the file is a PDF.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        # Ensure correct encoding, utf-8 is common
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

def process_file(file_path):
    """
    Processes one file: applies all chunking strategies (fixed, sentence, hybrid)
    with different sizes (256, 512, 1024), computes embeddings, adds chunk index,
    and stores all information.
    """
    source = os.path.basename(file_path)
    print(f"Processing file: {source}")
    text = load_raw_text(file_path)

    if text is None:
        print(f"Skipping file {source} due to loading error.")
        return

    # --- Generate Chunks for all methods and sizes ---
    print(f"  Generating fixed chunks...")
    chunks_fixed256 = chunk_fixed_256(text)
    chunks_fixed512 = chunk_fixed_512(text)
    chunks_fixed1024 = chunk_fixed_1024(text)

    print(f"  Generating sentence-aware chunks...")
    chunks_sentence_256 = sentence_aware_chunking(text, max_tokens=256)
    chunks_sentence_512 = sentence_aware_chunking(text, max_tokens=512)
    chunks_sentence_1024 = sentence_aware_chunking(text, max_tokens=1024)

    print(f"  Generating hybrid chunks...")
    chunks_hybrid_256 = hybrid_element_semantic_chunking(text, max_tokens=256)
    chunks_hybrid_512 = hybrid_element_semantic_chunking(text, max_tokens=512)
    chunks_hybrid_1024 = hybrid_element_semantic_chunking(text, max_tokens=1024)
    print(f"  Chunk generation complete.")

    # --- Map method names to chunk lists ---
    methods = {
        "fixed_256": chunks_fixed256,
        "fixed_512": chunks_fixed512,
        "fixed_1024": chunks_fixed1024,
        "sentence_aware_256": chunks_sentence_256,
        "sentence_aware_512": chunks_sentence_512,
        "sentence_aware_1024": chunks_sentence_1024,
        "hybrid_256": chunks_hybrid_256,
        "hybrid_512": chunks_hybrid_512,
        "hybrid_1024": chunks_hybrid_1024,
    }

    # --- Process and Store Chunks for each method ---
    for method, chunks in methods.items():
        # Skip if chunking failed or produced no chunks for this method
        if not chunks:
            print(f"Skipping method {method} as it produced no chunks.")
            continue

        print(f"Processing {len(chunks)} chunks for method: {method}")

        # Prepare chunk data WITH index
        chunk_data_with_index = []
        for i, chunk_text in enumerate(chunks):
            # Basic check for non-empty chunk text
            if chunk_text and chunk_text.strip():
                chunk_data_with_index.append({
                    "text": chunk_text.strip(), # Store stripped text
                    "index": i
                })
            else:
                 print(f"Warning: Skipping empty chunk at index {i} for method {method}")


        # Only proceed if there are valid chunks to store
        if chunk_data_with_index:
            # Pass the list of dictionaries to store_chunks
            # store_chunks now uses the imported embedding_processor.count_tokens internally
            store_chunks(
                chunks_with_meta=chunk_data_with_index,
                method=method,
                source=source,
                embedding_func=get_embedding
            )
        else:
            print(f"No valid chunks to store for method {method}.")


def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Build the absolute path to the data folder
    data_dir = os.path.join(script_dir, '..', 'data')
    # Debug: print the resolved data directory
    print("Looking for data files in:", data_dir)
    # List all files in the data directory
    data_files = glob.glob(os.path.join(data_dir, "*")) # Consider filtering for .pdf if other files exist
    print(f"Found data files: {data_files}")

    if not data_files:
        print(f"Warning: No files found in {data_dir}")
        return

    for file_path in data_files:
        # Simple check if it's a file (glob might return directories)
        if os.path.isfile(file_path):
            process_file(file_path)
        else:
            print(f"Skipping non-file item: {file_path}")

if __name__ == "__main__":
    # Get current time for context
    from datetime import datetime
    import pytz # For timezone handling: pip install pytz
    helsinki_tz = pytz.timezone('Europe/Helsinki')
    start_time = datetime.now(helsinki_tz)
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")

    main()

    end_time = datetime.now(helsinki_tz)
    print(f"Script finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print(f"Total execution time: {end_time - start_time}")