# src/main.py

import os
import glob
import re
import datetime
import time # Import time for timestamps
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase # Moved import here to group

# Import your project modules
from fixed_chunking import chunk_fixed_256, chunk_fixed_512, chunk_fixed_1024
from sentence_chunking import sentence_aware_chunking
from hybrid_chunking import hybrid_element_semantic_chunking
from embedding_processor import model as embedding_model
from pdf_extractor import extract_text_from_pdf
from neo4j_storage import store_chunks_neo4j

# --- Utility for timestamped logs ---
def log_info(message):
    """Prints an info message with a timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"INFO [{timestamp}]: {message}")

def log_process(message):
    """Prints a process step message with a timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"PROCESS [{timestamp}]: {message}")

# --- Load Environment Variables ---
log_info("Loading environment variables...")
load_dotenv(override=True)
log_info("Environment variables loaded.")

# --- Metadata Extraction Helper ---
def extract_doc_metadata(filename: str) -> Dict[str, Any]:
    """
    Extracts document metadata: filename, company, ticker, report type, quarter.
    Searches specifically for known tickers (AAPL, AMZN, INTC, MSFT, NVDA)
    anywhere in the filename to determine ticker and company name.
    """
    # Get just the filename part (e.g., "2023 Q2 AAPL.pdf") from a full path
    base_filename = os.path.basename(filename)

    # Initialize with defaults
    metadata = {
        "filename": base_filename, # Store the base filename
        "company_name": "Unknown",
        "ticker_symbol": "Unknown",
        "report_type": "10-Q",      # Default for Q1/Q2/Q3
        "fiscal_quarter_year": "Unknown",
    }

    # --- Start of Modified Logic for Ticker and Company Name ---

    # Define the known tickers and their corresponding company names
    # This map directly links the ticker found in the filename to the company name
    company_map = {
        "AAPL": "Apple Inc.",
        "AMZN": "Amazon.com, Inc.",
        "INTC": "Intel Corporation",
        "MSFT": "Microsoft Corporation",
        "NVDA": "NVIDIA Corporation",
    }

    # Try to find one of the known tickers anywhere in the filename
    found_ticker = None
    for ticker in company_map.keys(): # Iterate through "AAPL", "AMZN", etc.
        # Use word boundaries (\b) for exact match, case-insensitive search
        if re.search(r"\b" + re.escape(ticker) + r"\b", base_filename, re.IGNORECASE):
            found_ticker = ticker
            break # Found the ticker, no need to check others

    # If a known ticker was found in the filename, update metadata
    if found_ticker:
        metadata["ticker_symbol"] = found_ticker
        # Use the ticker found in the filename to get the company name from the map
        metadata["company_name"] = company_map[found_ticker]

    # --- End of Modified Logic ---

    # Fiscal quarter/year extraction logic (no changes needed here)
    # Searches for pattern like "2023 Q1", "2023_Q2", "2023Q3" etc.
    fq = re.search(r"(\d{4})[ _]?Q([1-3])", base_filename, re.IGNORECASE)
    if fq:
        year, q = fq.groups()
        metadata["fiscal_quarter_year"] = f"Q{q} {year}"
        # Note: report_type is already defaulted to 10-Q, which is correct for Q1-Q3.
        # If you might process 10-K (annual) reports later, you might need
        # additional logic here or based on filename patterns like "_10K_".

    return metadata

# --- Table Detection Helper ---
def check_if_table(chunk_text: str) -> bool:
    """
    Simple heuristic: if more than half the lines contain multiple numeric values, treat as a table.
    """
    lines = chunk_text.strip().split("\n")
    if len(lines) < 3:
        return False
    count = 0
    for line in lines:
        if len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", line)) > 2:
            count += 1
    return (count / len(lines)) > 0.5

# --- Core Processing ---

def load_raw_text(file_path: str) -> Optional[str]:
    log_process(f"Attempting to load raw text from: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    text_content = None
    try:
        if ext == ".pdf":
            text_content = extract_text_from_pdf(file_path)
        else: # Assuming .txt or similar
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        if text_content:
            log_process(f"Successfully loaded text (length: {len(text_content)}) from: {file_path}")
        else:
            log_process(f"No text content found or extracted from: {file_path}")
        return text_content
    except Exception as e:
        log_info(f"ERROR loading text from {file_path}: {e}")
        return None


def process_file(file_path: str, session) -> None:
    log_process(f"--- Starting processing file: {os.path.basename(file_path)} ---")
    start_time = time.time()

    text = load_raw_text(file_path)
    if not text:
        log_info(f"Skipping empty or unreadable file: {file_path}")
        print("-" * 40) # Separator
        return

    log_process("Extracting document metadata...")
    doc_meta = extract_doc_metadata(os.path.basename(file_path))
    log_process(f"Metadata extracted: {doc_meta}")

    log_process("Applying chunking strategies...")
    methods = {}
    try:
        # Wrap chunking calls in try-except for robustness? Optional.
        methods = {
            "fixed_256":          chunk_fixed_256(text),
            "fixed_512":          chunk_fixed_512(text),
            "fixed_1024":         chunk_fixed_1024(text),
            "sentence_aware_256": sentence_aware_chunking(text, max_tokens=256),
            "sentence_aware_512": sentence_aware_chunking(text, max_tokens=512),
            "sentence_aware_1024":sentence_aware_chunking(text, max_tokens=1024),
            "hybrid_256":         hybrid_element_semantic_chunking(text, max_tokens=256),
            "hybrid_512":         hybrid_element_semantic_chunking(text, max_tokens=512),
            "hybrid_1024":        hybrid_element_semantic_chunking(text, max_tokens=1024),
        }
        log_process("Chunking strategies applied.")
    except Exception as e:
        log_info(f"ERROR during chunking for file {file_path}: {e}")
        print("-" * 40) # Separator
        return # Skip rest of processing for this file if chunking fails

    total_chunks_stored_for_file = 0
    for method, chunks in methods.items():
        method_start_time = time.time()
        log_process(f"  Processing method: {method}")
        if not chunks:
            log_process(f"    No chunks generated by method {method}, skipping.")
            continue

        log_process(f"    Found {len(chunks)} potential chunks for method {method}.")
        metas, texts = [], []
        # This inner loop iterates through individual chunks - avoid printing here unless debugging deeply
        for i, chunk in enumerate(chunks):
            c = chunk.strip()
            if not c:
                continue # Skip empty chunks after stripping
            metas.append({
                **doc_meta,
                "chunk_id":        f"{doc_meta['filename']}_{method}_{i}",
                "chunk_index":     i,
                "chunking_method": method,
                "is_table":        check_if_table(c),
                "text":            c # Store the stripped text
            })
            texts.append(c) # Use stripped text for embedding

        if not texts:
            log_process(f"    No non-empty chunks found for method {method} after stripping, skipping embedding/storage.")
            continue

        try:
            log_process(f"    Generating embeddings for {len(texts)} chunks...")
            embedding_start_time = time.time()
            # Ensure embedding_model is not None (should be handled by embedding_processor import)
            if embedding_model is None:
                 log_info(f"ERROR: Embedding model is None. Cannot generate embeddings for method {method}.")
                 continue # Skip to next method
            embeddings = embedding_model.encode(texts, normalize_embeddings=True).tolist()
            embedding_end_time = time.time()
            log_process(f"    Embeddings generated in {embedding_end_time - embedding_start_time:.2f} seconds.")

            log_process(f"    Storing {len(metas)} chunks and embeddings for method {method}...")
            storage_start_time = time.time()
            store_chunks_neo4j(session, metas, embeddings) # Assumes this function handles internal errors/logging
            storage_end_time = time.time()
            log_process(f"    Chunks stored in {storage_end_time - storage_start_time:.2f} seconds.")
            total_chunks_stored_for_file += len(metas)

        except Exception as e:
            log_info(f"ERROR during embedding or storage for method {method} on file {file_path}: {e}")
            # Decide whether to continue with other methods or stop processing this file

        method_end_time = time.time()
        log_process(f"  Finished processing method {method} in {method_end_time - method_start_time:.2f} seconds.")

    end_time = time.time()
    log_process(f"--- Finished processing file: {os.path.basename(file_path)} in {end_time - start_time:.2f} seconds. Total chunks stored: {total_chunks_stored_for_file} ---")
    print("-" * 40) # Separator

def main():
    log_info("--- Script execution started ---")
    start_time = time.time()

    # Neo4j Connection Details
    uri      = os.getenv("NEO4J_URI")
    user     = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    log_info(f"Attempting connection to Neo4j URI: {uri}")
    if not uri or not user or not password:
         log_info("ERROR: Neo4j connection details missing in environment variables. Exiting.")
         return

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity() # Check if connection is initially possible
        log_info("Successfully connected to Neo4j.")
    except Exception as e:
        log_info(f"ERROR: Failed to connect to Neo4j at {uri}: {e}")
        return # Exit if connection fails

    # File Processing
    data_path = os.path.join("data", "*.pdf")
    log_info(f"Searching for PDF files in: {data_path}")
    file_list = glob.glob(data_path)
    log_info(f"Found {len(file_list)} PDF files to process.")

    if not file_list:
        log_info("No files found to process. Exiting.")
    else:
        try:
            # Use driver context manager for session handling
            with driver.session() as session:
                log_info("Neo4j session opened.")
                for i, fp in enumerate(file_list):
                    log_info(f"Processing file {i+1}/{len(file_list)}: {os.path.basename(fp)}")
                    process_file(fp, session) # Pass the active session
                log_info("Finished processing all files.")
        except Exception as e:
            log_info(f"ERROR during file processing loop or session management: {e}")
        finally:
            # Close Driver
            if driver:
                log_info("Closing Neo4j driver connection...")
                driver.close()
                log_info("Neo4j driver closed.")

    end_time = time.time()
    log_info(f"--- Script execution finished in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()