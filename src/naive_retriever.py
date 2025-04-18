# src/naive_retriever.py
import os, textwrap, requests
from neo4j import GraphDatabase
from embedding_processor import get_embedding 

INDEX_NAME   = "chunk_bge_embeddings"          # vector index in Neo4j
EMBED_MODEL  = "bge-base-en-v1.5"              # stays in sync with your chunks

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

def top_k_chunks(question: str, k: int = 5):
    """Return top‑k {text,score} dicts (naïve vector search only)."""
    q_emb = get_embedding("Represent this sentence for searching relevant passages: " + question)
    cypher = """
    CALL db.index.vector.queryNodes($idx,$k,$vec)
    YIELD node, score
    WHERE node.embedding_model = $model
    RETURN node.text AS text, score
    ORDER BY score DESC
    """
    with driver.session() as s:
        return s.run(cypher, idx=INDEX_NAME, k=k, vec=q_emb, model=EMBED_MODEL).data()

def naive_context(question: str, k: int = 5):
    hits = top_k_chunks(question, k)
    ctx  = "\n\n---\n\n".join(h["text"] for h in hits)
    return ctx, hits            # second item is metadata

if __name__ == "__main__":      # quick smoke‑test
    print(naive_context("What is Apple’s iPhone revenue?", k=5)[0][:400])
