# src/neo4j_storage.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "your_password")

driver = GraphDatabase.driver(uri, auth=(username, password))

def store_document(tx, source):
    """
    Creates or finds a Document node for the source file.
    """
    query = """
    MERGE (d:Document {source: $source})
    ON CREATE SET d.created = timestamp()
    RETURN id(d) as doc_id
    """
    result = tx.run(query, source=source)
    record = result.single()
    return record["doc_id"] if record else None

def store_chunk_with_relationship(tx, chunk_text, method, source, token_count, embedding, embedding_cost):
    """
    Creates a Chunk node with extra metadata and links it to its Document node.
    """
    query = """
    MERGE (d:Document {source: $source})
      ON CREATE SET d.created = timestamp()
    CREATE (c:Chunk {
        text: $text,
        method: $method,
        token_count: $token_count,
        embedding: $embedding,
        embedding_cost: $embedding_cost
    })
    CREATE (d)-[:CONTAINS]->(c)
    RETURN id(c) as id
    """
    result = tx.run(query, text=chunk_text, method=method, source=source,
                    token_count=token_count, embedding=embedding, embedding_cost=embedding_cost)
    record = result.single()
    return record["id"] if record else None

def store_chunks(chunks, method, source, embedding_func, tokenizer):
    """
    Loops over chunks, computes token counts, embeddings, cost estimates, and stores each chunk in Neo4j with relationships.
    """
    from embedding_processor import compute_embedding_cost
    with driver.session() as session:
        for chunk in chunks:
            token_count = len(tokenizer.encode(chunk))
            embedding = embedding_func(chunk)
            cost = compute_embedding_cost(token_count)
            node_id = session.write_transaction(store_chunk_with_relationship,
                                                  chunk, method, source, token_count, embedding, cost)
            print(f"Stored chunk from {source} using {method} with node id: {node_id}")
