# src/neo4j_storage.py

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import List, Dict, Any
from embedding_processor import count_tokens # Ensure this is correctly imported

# Load environment and initialize driver (assuming this part is fine)
dotenv_loaded = load_dotenv()
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "password")

# It might be better to manage the driver instance within your main script
# or use a dedicated connection manager, but we'll keep this structure for now.
# Consider passing the driver or session into functions instead of using a global.
driver = None
try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    # You might want to verify connectivity once at the start of main.py
except Exception as e:
    print(f"ERROR: Failed to create Neo4j driver: {e}")
    # Handle driver creation failure appropriately

def store_chunks_neo4j(session, metas: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Stores Document and Chunk nodes with full metadata, token counts, embeddings,
    and heading information.
    """
    # Check if session is valid (basic check)
    if not hasattr(session, 'run'):
        print("ERROR: Invalid Neo4j session passed to store_chunks_neo4j")
        return

    for meta, emb in zip(metas, embeddings):
        # Compute token count
        tok_cnt = count_tokens(meta['text'])

        # --- MODIFIED SECTION START ---
        # Build params including all desired properties, ADDING 'heading'
        params: Dict[str, Any] = {
            'filename':            meta['filename'],
            'company_name':        meta['company_name'],
            'ticker_symbol':       meta['ticker_symbol'],
            'report_type':         meta['report_type'],
            'fiscal_quarter_year': meta['fiscal_quarter_year'],
            'chunk_id':            meta['chunk_id'],
            'chunk_index':         meta['chunk_index'],
            'method':              meta['chunking_method'],
            'is_table':            meta['is_table'],
            'heading':             meta.get('heading'), # Use .get() for safety if heading might be missing
            'token_count':         tok_cnt,
            'text':                meta['text'],
            'embedding':           emb
        }

        # Merge document node and set metadata
        # Merge chunk node and set properties, ADDING 'heading'
        try:
            session.run(
                '''
                MERGE (d:Document {filename: $filename})
                SET d.company_name        = $company_name,
                    d.ticker_symbol       = $ticker_symbol,
                    d.report_type         = $report_type,
                    d.fiscal_quarter_year = $fiscal_quarter_year

                MERGE (c:Chunk {id: $chunk_id})
                SET c.text         = $text,
                    c.chunk_index  = $chunk_index,
                    c.method       = $method,
                    c.is_table     = $is_table,
                    c.heading      = $heading, // <-- ADDED HEADING HERE
                    c.token_count  = $token_count,
                    c.embedding    = $embedding

                MERGE (d)-[:CONTAINS]->(c)
                ''',
                params
            )
        except Exception as e:
             # Add some error logging for individual chunk storage failures
             print(f"ERROR storing chunk {params.get('chunk_id', 'N/A')}: {e}")
             # Decide if you want to continue or stop processing on error
