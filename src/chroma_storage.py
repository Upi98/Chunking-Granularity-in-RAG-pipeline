# src/chroma_storage.py
import chromadb
import os
from typing import List, Dict, Any

# --- Configuration ---
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "document_chunks")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" # Consistent model name

# --- Initialize ChromaDB Client ---
try:
    print(f"DEBUG: Initializing ChromaDB PersistentClient at path: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    print("DEBUG: ChromaDB client initialized successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize ChromaDB client: {e}")
    client = None

# --- Collection Management ---
def get_or_create_collection(collection_name=COLLECTION_NAME):
    """Gets or creates a ChromaDB collection."""
    if client is None:
        print("ERROR: ChromaDB client not initialized. Cannot get/create collection.")
        return None
    try:
        print(f"DEBUG: Getting or creating ChromaDB collection: '{collection_name}'")
        collection = client.get_or_create_collection(name=collection_name)
        print(f"DEBUG: Collection '{collection_name}' ready.")
        return collection
    except Exception as e:
        print(f"ERROR: Failed to get or create collection '{collection_name}': {e}")
        return None

# --- Storage Function ---
def store_chunks_chroma(collection, chunks_with_meta: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Stores chunks with their metadata (including enhanced financial fields) and embeddings
    in the specified ChromaDB collection. Uses upsert for idempotency.
    """
    if collection is None:
        print("ERROR: ChromaDB collection object is None. Cannot store chunks.")
        return

    if not chunks_with_meta or not embeddings or len(chunks_with_meta) != len(embeddings):
        print("ERROR: Invalid input for store_chunks_chroma. Ensure chunks and embeddings lists are non-empty and have matching lengths.")
        return

    ids_to_store = []
    documents_to_store = []
    metadata_to_store = []

    print(f"DEBUG: Preparing {len(chunks_with_meta)} chunks for ChromaDB upsert...")
    for i, chunk_data in enumerate(chunks_with_meta):
        # --- Basic Info ---
        source = chunk_data.get("source_filename", "unknown_source") # Use consistent key
        method = chunk_data.get("chunking_method", "unknown_method") # Use consistent key
        chunk_index = chunk_data.get("chunk_index", -1) # Use consistent key
        chunk_id = f"{source}_{method}_{chunk_index}" # Unique ID

        ids_to_store.append(chunk_id)
        documents_to_store.append(chunk_data["text"]) # The actual text content

        # --- Prepare Metadata ---
        # Include all fields passed in chunk_data, providing defaults for safety
        metadata = {
            # --- Basic Metadata ---
            "source_filename": source,
            "chunking_method": method,
            "chunk_index": int(chunk_index),
            "token_count": int(chunk_data.get("token_count", 0)),
            "embedding_model": chunk_data.get("embedding_model", EMBEDDING_MODEL_NAME),
            # --- Document Level Metadata ---
            "company_name": chunk_data.get("company_name", "Unknown"),
            "ticker_symbol": chunk_data.get("ticker_symbol", "Unknown"),
            "report_type": chunk_data.get("report_type", "Unknown"),
            "period_end_date": chunk_data.get("period_end_date", "Unknown"),
            "filing_date": chunk_data.get("filing_date", "Unknown"),
            "fiscal_quarter_year": chunk_data.get("fiscal_quarter_year", "Unknown"),
            # --- Chunk Level Metadata ---
            "page_number": int(chunk_data.get("page_number", -1)), # Store as int, -1 if unknown
            "section_title": chunk_data.get("section_title", "Unknown"),
            "is_table_data": bool(chunk_data.get("is_table_data", False)), # Store as bool
            # --- Optional Advanced ---
            # "mentioned_products_services": chunk_data.get("mentioned_products_services", []), # Store as list
            # "preceding_chunk_id": chunk_data.get("preceding_chunk_id", ""),
            # "succeeding_chunk_id": chunk_data.get("succeeding_chunk_id", "")
        }
        # Ensure all metadata values are Chroma-compatible (str, int, float, bool)
        # Lists of strings are generally okay too.
        for key, value in metadata.items():
             if not isinstance(value, (str, int, float, bool, list)):
                 print(f"Warning: Metadata key '{key}' has non-standard type '{type(value)}' for chunk {chunk_id}. Converting to string.")
                 metadata[key] = str(value)

        metadata_to_store.append(metadata)

    # Perform the upsert operation
    try:
        print(f"DEBUG: Upserting {len(ids_to_store)} items into collection '{collection.name}'...")
        collection.upsert(
            ids=ids_to_store,
            embeddings=embeddings,
            documents=documents_to_store,
            metadatas=metadata_to_store
        )
        print(f"DEBUG: Successfully upserted {len(ids_to_store)} items.")
    except Exception as e:
        print(f"ERROR: Failed to upsert data into ChromaDB collection '{collection.name}': {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging