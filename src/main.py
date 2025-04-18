# src/main.py

import os
import glob
import re
import datetime # For date parsing/formatting
from typing import List, Dict, Any, Optional # Added for type hinting
from fixed_chunking import chunk_fixed_256, chunk_fixed_512, chunk_fixed_1024
from sentence_chunking import sentence_aware_chunking
from hybrid_chunking import hybrid_element_semantic_chunking
from embedding_processor import count_tokens
from chroma_storage import get_or_create_collection, store_chunks_chroma # Import Chroma functions
# Make sure pdf_extractor is available and returns full text
# Ideally, modify pdf_extractor to return List[Tuple[int, str]] (page_number, page_text)
from pdf_extractor import extract_text_from_pdf
# Import the model directly for batch embedding if available
from embedding_processor import model as embedding_model

# --- Define Embedding Model Name Constant HERE ---
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# --- Helper Functions for Basic Metadata Extraction ---
# (Keep the helper functions extract_doc_metadata, find_chunk_section, check_if_table as defined in the previous response)

def extract_doc_metadata(filename: str, text_start: str, text_end: str) -> Dict[str, Any]:
    """Extracts document-level metadata from filename and text snippets."""
    metadata = {
        "company_name": "Unknown",
        "ticker_symbol": "Unknown",
        "report_type": "Unknown",
        "period_end_date": "Unknown",
        "filing_date": "Unknown",
        "fiscal_quarter_year": "Unknown",
    }
    # Company Name (Often explicitly stated)
    if "Apple Inc." in text_start[:1000]: # Check first few lines
         metadata["company_name"] = "Apple Inc."
    # Ticker Symbol (From filename for AAPL)
    if "AAPL" in filename:
        metadata["ticker_symbol"] = "AAPL"
    # Report Type (From filename or text)
    if "10-Q" in filename or "FORM 10-Q" in text_start[:200]:
         metadata["report_type"] = "10-Q"
    elif "10-K" in filename or "FORM 10-K" in text_start[:200]:
         metadata["report_type"] = "10-K"
    # Period End Date (Regex on cover page info)
    period_match = re.search(r'For the quarterly period ended (\w+\s+\d{1,2},\s+\d{4})', text_start[:1000])
    if period_match:
        try:
            date_obj = datetime.datetime.strptime(period_match.group(1), '%B %d, %Y')
            metadata["period_end_date"] = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            metadata["period_end_date"] = period_match.group(1)
    else: # Fallback from filename
         fn_match = re.search(r'(\d{4})\s+Q(\d)', filename)
         if fn_match:
             year, quarter = fn_match.groups()
             if quarter == '1': metadata["period_end_date"] = f"{int(year)-1}-12-31" # Apple Q1 ends Dec 31 of previous year
             elif quarter == '2': metadata["period_end_date"] = f"{year}-03-31" # Approx end of March (needs checking fiscal cal)
             elif quarter == '3': metadata["period_end_date"] = f"{year}-06-30" # Approx end of June (needs checking fiscal cal)
             elif quarter == '4': metadata["period_end_date"] = f"{year}-09-30" # Approx end of Sept
    # Filing Date (Regex near signature/end)
    filing_match = re.search(r'Date:\s+(\w+\s+\d{1,2},\s+\d{4})', text_end[-1000:]) # Check last 1000 chars
    if filing_match:
         try:
            date_obj = datetime.datetime.strptime(filing_match.group(1), '%B %d, %Y')
            metadata["filing_date"] = date_obj.strftime('%Y-%m-%d')
         except ValueError:
            metadata["filing_date"] = filing_match.group(1)
    # Fiscal Quarter/Year (From filename)
    fn_match_fq = re.search(r'(\d{4})\s+Q(\d)', filename)
    if fn_match_fq:
        year, quarter = fn_match_fq.groups()
        metadata["fiscal_quarter_year"] = f"Q{quarter} {year}"
    elif metadata["period_end_date"] != "Unknown": # Try deriving from period end date
         try:
             dt = datetime.datetime.strptime(metadata["period_end_date"], '%Y-%m-%d')
             month = dt.month
             year = dt.year # Calendar year of period end
             # Apple Fiscal Year mapping (ends Sep)
             if month in [10, 11, 12]: quarter, f_year = "Q1", year + 1
             elif month in [1, 2, 3]: quarter, f_year = "Q2", year
             elif month in [4, 5, 6]: quarter, f_year = "Q3", year
             elif month in [7, 8, 9]: quarter, f_year = "Q4", year
             else: quarter, f_year = "?", year
             metadata["fiscal_quarter_year"] = f"{quarter} {f_year}" # Use Fiscal Year
         except: pass
    # Clean up potential company name from filename
    if metadata["company_name"] == "Unknown" and metadata["ticker_symbol"] != "Unknown":
         base = os.path.splitext(filename)[0]
         base = base.replace(metadata["ticker_symbol"], "").strip()
         base = re.sub(r'\d{4}\s+Q\d', '', base).strip()
         metadata["company_name"] = base if base else "Unknown"
    return metadata

def find_chunk_section(chunk_text: str) -> str:
    # (Keep existing implementation)
    lines = chunk_text.split('\n', 5)
    for line in lines:
        clean_line = line.strip()
        match_item = re.match(r'^(Item\s+\d{1,2}A?\.?|Note\s+\d{1,2}\.?)\s*(.*)', clean_line, re.IGNORECASE)
        if match_item:
            title_part = match_item.group(2).strip().split('\n')[0] # Take only first line of title
            return f"{match_item.group(1).upper().strip('.')} {title_part[:100]}".strip()
        if re.match(r"Management's Discussion and Analysis", clean_line, re.IGNORECASE):
            return "Management's Discussion and Analysis of Financial Condition and Results of Operations"
        # Add other common sections...
        if re.match(r"Financial Statements", clean_line, re.IGNORECASE): return "Financial Statements"
        if re.match(r"Risk Factors", clean_line, re.IGNORECASE): return "Risk Factors"
        if re.match(r"Legal Proceedings", clean_line, re.IGNORECASE): return "Legal Proceedings"
        if re.match(r"Controls and Procedures", clean_line, re.IGNORECASE): return "Controls and Procedures"
        if re.match(r"Quantitative and Qualitative Disclosures", clean_line, re.IGNORECASE): return "Quantitative and Qualitative Disclosures About Market Risk"
    return "Unknown"

def check_if_table(chunk_text: str) -> bool:
    # (Keep existing implementation)
    lines = chunk_text.strip().split('\n')
    if len(lines) < 3: return False
    numeric_lines = 0
    dollar_signs = chunk_text.count('$')
    percentage_signs = chunk_text.count('%')
    for line in lines:
        if len(re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\(\s*\d[\d,.]*\s*\)', line)) > 2:
             numeric_lines += 1
    if (numeric_lines / len(lines) > 0.5) or (dollar_signs > len(lines) / 2) or (percentage_signs > len(lines) / 4):
        if any(kw in chunk_text for kw in ["Three Months Ended", "Nine Months Ended", "Six Months Ended", "Assets", "Liabilities", "Shareholders' Equity"]): return True
        if "% Change" in chunk_text or "increase/(decrease)" in chunk_text.lower(): return True
        return True
    return False

# --- Core Processing Functions ---
def load_raw_text(file_path):
    # (Keep existing implementation)
    print(f"DEBUG: Loading full text from {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    # (Keep existing implementation)
    if embedding_model is None:
        print("ERROR: Embedding model not loaded.")
        return [None] * len(texts)
    if not texts: return []
    try:
        print(f"DEBUG: Generating embeddings for batch of {len(texts)} texts...")
        embeddings_np = embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        print(f"DEBUG: Embedding generation complete.")
        return embeddings_np.tolist()
    except Exception as e:
        print(f"ERROR: Batch embedding generation failed: {e}")
        return [None] * len(texts)

def process_file(file_path, collection):
    """
    Processes one file: extracts metadata, applies chunking, computes embeddings,
    and stores all information in ChromaDB.
    """
    if collection is None:
        print(f"ERROR: Cannot process file {file_path}, ChromaDB collection is not available.")
        return

    source_filename = os.path.basename(file_path)
    print(f"Processing file: {source_filename}")
    full_text = load_raw_text(file_path)

    if full_text is None or not full_text.strip():
        print(f"Skipping file {source_filename} due to loading error or empty content.")
        return

    # Extract Document Level Metadata
    text_start_snippet = full_text[:2000]
    text_end_snippet = full_text[-2000:]
    doc_metadata = extract_doc_metadata(source_filename, text_start_snippet, text_end_snippet)
    print(f"  Extracted Doc Metadata: {doc_metadata}")

    # Generate Chunks
    print(f"  Generating chunks...")
    methods = {
        "fixed_256": chunk_fixed_256(full_text),
        "fixed_512": chunk_fixed_512(full_text),
        "fixed_1024": chunk_fixed_1024(full_text),
        "sentence_aware_256": sentence_aware_chunking(full_text, max_tokens=256),
        "sentence_aware_512": sentence_aware_chunking(full_text, max_tokens=512),
        "sentence_aware_1024": sentence_aware_chunking(full_text, max_tokens=1024),
        "hybrid_256": hybrid_element_semantic_chunking(full_text, max_tokens=256),
        "hybrid_512": hybrid_element_semantic_chunking(full_text, max_tokens=512),
        "hybrid_1024": hybrid_element_semantic_chunking(full_text, max_tokens=1024),
    }
    print(f"  Chunk generation complete.")

    # Process and Store Chunks for each method
    for method, chunks_list in methods.items():
        if not chunks_list:
            print(f"Skipping method {method} for {source_filename} as it produced no chunks.")
            continue

        print(f"Processing {len(chunks_list)} chunks for method: {method} from {source_filename}")

        valid_chunk_texts = []
        chunks_with_meta = []
        num_chunks_for_method = len(chunks_list) # Total chunks for succeeding_id check

        for i, chunk_text in enumerate(chunks_list):
            stripped_text = chunk_text.strip() if chunk_text else ""
            if stripped_text:
                token_count = count_tokens(stripped_text)
                section_title = find_chunk_section(stripped_text)
                is_table = check_if_table(stripped_text)
                # page_number placeholder

                # Combine metadata
                meta = {
                    **doc_metadata, # Add document level fields
                    "text": stripped_text, # Include text temporarily for potential use
                    "chunk_index": i,
                    "source_filename": source_filename,
                    "chunking_method": method,
                    "token_count": token_count,
                    "embedding_model": EMBEDDING_MODEL_NAME, # Use constant defined in this file
                    "section_title": section_title,
                    "is_table_data": is_table,
                    "page_number": -1, # Placeholder
                    "preceding_chunk_id": f"{source_filename}_{method}_{i-1}" if i > 0 else "",
                    # Set succeeding_id only if it's not the last chunk
                    "succeeding_chunk_id": f"{source_filename}_{method}_{i+1}" if i < num_chunks_for_method - 1 else "",
                }
                # Remove the temporary text field before storing if desired
                # meta.pop("text", None)
                # Although store_chunks_chroma doesn't strictly need it removed

                chunks_with_meta.append(meta)
                valid_chunk_texts.append(stripped_text)
            else:
                print(f"Warning: Skipping empty chunk at index {i} for method {method}")

        # Embedding and Storage
        if not valid_chunk_texts:
            print(f"No valid chunks to process for method {method} from {source_filename}.")
            continue

        embeddings = generate_embeddings_batch(valid_chunk_texts)

        # Filter out chunks where embedding failed
        final_chunks_to_store = []
        final_embeddings_to_store = []
        if len(chunks_with_meta) == len(embeddings):
            for i, emb in enumerate(embeddings):
                if emb is not None:
                    final_chunks_to_store.append(chunks_with_meta[i])
                    final_embeddings_to_store.append(emb)
                else:
                    print(f"Warning: Skipping chunk {chunks_with_meta[i]['chunk_index']} for method {method} due to embedding error.")
        else:
            print(f"ERROR: Mismatch between chunk count ({len(chunks_with_meta)}) and embedding count ({len(embeddings)}) for method {method}. Skipping storage.")
            continue

        # Store the batch
        if final_chunks_to_store and final_embeddings_to_store:
            print(f"Storing {len(final_chunks_to_store)} chunks with full metadata for method {method}...")
            # Pass the list containing metadata dicts (which includes 'text' field temporarily)
            store_chunks_chroma(collection, final_chunks_to_store, final_embeddings_to_store)
        else:
            print(f"No valid chunks with embeddings to store for method {method}.")


# --- Main Execution Block ---
def main():
    # (Keep existing implementation)
    collection = get_or_create_collection()
    if collection is None:
        print("FATAL: Could not get or create ChromaDB collection. Exiting.")
        return

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
    print("Looking for data files in:", data_dir)
    data_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    print(f"Found PDF files: {data_files}")

    if not data_files:
        print(f"Warning: No PDF files found in {data_dir}")
        return

    for file_path in data_files:
        if os.path.isfile(file_path):
            process_file(file_path, collection)
        else:
            print(f"Skipping non-file item: {file_path}")

if __name__ == "__main__":
    # (Keep existing implementation, consider standard datetime)
    start_time = datetime.datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = datetime.datetime.now()
    print(f"Script finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {end_time - start_time}")