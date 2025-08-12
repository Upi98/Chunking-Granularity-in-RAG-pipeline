# src/neo4j_storage.py

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import List, Dict, Any
from embedding_processor import count_tokens

# Load environment and initialize driver
dotenv_loaded = load_dotenv()
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(uri, auth=(user, password))

def store_chunks_neo4j(session, metas: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Stores Document and Chunk nodes with full metadata, token counts, and embeddings.
    """
    for meta, emb in zip(metas, embeddings):
        # Compute token count
        tok_cnt = count_tokens(meta['text'])
        # Build params including all desired properties
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
            'token_count':         tok_cnt,
            'text':                meta['text'],
            'embedding':           emb
        }
        # Merge document node and set metadata
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
                c.token_count  = $token_count,
                c.embedding    = $embedding

            MERGE (d)-[:CONTAINS]->(c)
            ''',
            params
        )