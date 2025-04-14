# src/neo4j_storage.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from embedding_processor import count_tokens

load_dotenv()
print("DEBUG: .env loaded (presumably)") # Check if this line appears

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "your_password") 


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

def store_chunk_with_relationship(tx, chunk_text, chunk_index, method, source, token_count, embedding):
    """
    Stores a chunk node with properties including chunk_index and embedding_model,
    and links it to the source Document node. Cost is removed.
    """
    embedding_model_name = "bge-base-en-v1.5" # Define BGE model name

    query = """
    MERGE (d:Document {source: $source}) ON CREATE SET d.created = timestamp()
    CREATE (c:Chunk {
        text: $text,
        chunk_index: $chunk_index, // <-- Store the index
        method: $method,
        token_count: $token_count,
        embedding: $embedding,
        source: $source,
        embedding_model: $embedding_model_name
    })
    CREATE (d)-[:CONTAINS]->(c)
    RETURN elementId(c) as id
    """
    result = tx.run(query,
                    text=chunk_text,
                    chunk_index=chunk_index, # Pass index parameter
                    method=method,
                    source=source,
                    token_count=token_count,
                    embedding=embedding,
                    embedding_model_name=embedding_model_name)
    record = result.single()
    return record["id"] if record else None

# Modify to accept list of dicts and pass index
def store_chunks(chunks_with_meta, method, source, embedding_func): # Renamed param
    """
    Loops over chunks (now dicts with 'text' and 'index'), computes token counts,
    embeddings, and stores each chunk in Neo4j with relationships and index.
    """
    if driver is None:
        print("ERROR: Neo4j driver not initialized in store_chunks. Aborting.")
        return

    with driver.session() as session:
        # Iterate through the list of dictionaries
        for chunk_data in chunks_with_meta:
            chunk_text = chunk_data["text"]
            chunk_index = chunk_data["index"] # Get the index

            # Use the imported count_tokens function for consistency
            token_count = count_tokens(chunk_text) # Use BGE tokenizer count

            embedding = embedding_func(chunk_text)

            if embedding is not None:
                node_id = session.write_transaction(
                    store_chunk_with_relationship,
                    chunk_text,
                    chunk_index, # Pass the index
                    method,
                    source,
                    token_count,
                    embedding
                )
                if node_id:
                     print(f"Stored chunk {chunk_index} from {source} using {method} ({token_count} BGE tokens) with node id: {node_id}")
                else:
                     print(f"Failed to store chunk {chunk_index} from {source} using {method} (tx returned None)")
            else:
                print(f"Skipped storing chunk {chunk_index} from {source} using {method} due to embedding error.")
