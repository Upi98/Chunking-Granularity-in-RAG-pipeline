# src/neo4j_storage.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
print("DEBUG: .env loaded (presumably)") # Check if this line appears

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "your_password") # Default if not in .env

# --- VERIFY THESE DEBUG PRINTS ARE PRESENT ---
print(f"DEBUG: Attempting to connect with:")
print(f"DEBUG:   URI: {uri}")
print(f"DEBUG:   Username: {username}")
# Mask password for security, but show length
password_display = f"{password[:1]}...{password[-1:]}" if password and len(password) > 1 else ("<empty>" if not password else password)
print(f"DEBUG:   Password (masked): {password_display} (Length: {len(password) if password else 0})")
# --- END DEBUG PRINTS ---

# Initialize driver (inside a try block is good practice)
driver = None # Initialize driver as None
try:
    if uri and username and password: # Check if credentials were loaded
        driver = GraphDatabase.driver(uri, auth=(username, password))
        # Optional: Verify connectivity immediately after creating driver
        # print("DEBUG: Attempting driver verification...")
        # driver.verify_connectivity()
        # print("DEBUG: Driver verification successful.")
        print("DEBUG: Neo4j driver object initialized.")
    else:
        print("ERROR: Missing Neo4j URI, Username, or Password in environment/.env file.")

except Exception as e:
    print(f"ERROR connecting to Neo4j or initializing driver: {e}")
    # driver remains None

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
    Stores a chunk node with its properties (including source)
    and links it to the source Document node.
    """
    query = """
    // Ensure the Document node exists
    MERGE (d:Document {source: $source})
      ON CREATE SET d.created = timestamp()
    // Create the Chunk node with properties, INCLUDING source
    CREATE (c:Chunk {
        text: $text,
        method: $method,
        token_count: $token_count,
        embedding: $embedding,
        embedding_cost: $embedding_cost,
        source: $source  
    })
    // Create the relationship
    CREATE (d)-[:CONTAINS]->(c)
    RETURN elementId(c) as id
    """
    # Parameters passed remain the same
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
