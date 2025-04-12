import os
import glob
import tiktoken
from fixed_chunking import chunk_fixed_256, chunk_fixed_512, chunk_fixed_1024
from sentence_chunking import sentence_aware_chunking
from hybrid_chunking import hybrid_element_semantic_chunking
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
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def process_file(file_path):
    """
    Processes one file: applies all chunking strategies, computes embeddings, and stores all information.
    """
    source = os.path.basename(file_path)
    print(f"Processing file: {source}")
    text = load_raw_text(file_path)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    chunks_fixed256 = chunk_fixed_256(text)
    chunks_fixed512 = chunk_fixed_512(text)
    chunks_fixed1024 = chunk_fixed_1024(text)
    chunks_sentence = sentence_aware_chunking(text, max_tokens=512)
    chunks_hybrid = hybrid_element_semantic_chunking(text, max_tokens=1024)

    methods = {
        "fixed_256": chunks_fixed256,
        "fixed_512": chunks_fixed512,
        "fixed_1024": chunks_fixed1024,
        "sentence_aware": chunks_sentence,
        "hybrid": chunks_hybrid
    }

    for method, chunks in methods.items():
        print(f"Processing {len(chunks)} chunks for method: {method}")
        store_chunks(chunks, method=method, source=source, embedding_func=get_embedding, tokenizer=tokenizer)

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Build the absolute path to the data folder
    data_dir = os.path.join(script_dir, '..', 'data')
    # Debug: print the resolved data directory
    print("Looking for data files in:", data_dir)
    # List all files in the data directory
    data_files = glob.glob(os.path.join(data_dir, "*"))
    print("Found data files:", data_files)
    
    for file_path in data_files:
        process_file(file_path)
    
if __name__ == "__main__":
    main()