# src/pathrag_retriever.py

import os
from neo4j import GraphDatabase
from embedding_processor import get_embedding, count_tokens # Use the BGE embedding function and tokenizer count
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

# --- Neo4j Connection (Can be passed or initialized here) ---
# It's often better to pass the driver instance
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
NEO4J_DATABASE = "neo4j"


# --- Constants ---
BGE_VECTOR_INDEX_NAME = "chunk_bge_embeddings" # Make sure this matches your index name
BGE_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1" # To filter for correct chunks
ENTITY_LABELS = ["Company", "Quarter", "Person", "Location", "Organization"] # Labels created by the enrichment script

def format_context(results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Formats the retrieved chunk texts into a single context string,
    handling duplicates and sorting.
    Returns the context string and a list of metadata dicts for the used chunks.
    """
    unique_chunks = {}  # Use dict to deduplicate based on elementId
    for record in results:
        chunk_id = record.get("chunk_id")
        # Ensure basic data is present
        if chunk_id and record.get("text") is not None and chunk_id not in unique_chunks:
            unique_chunks[chunk_id] = {
                "id": chunk_id,
                "text": record.get("text", ""),
                "source": record.get("source", "Unknown"),
                "method": record.get("method", "Unknown"),
                # Default index to a large number if missing for sorting robustness
                "index": record.get("index") if record.get("index") is not None else float('inf'),
                "score": record.get("score") # Will be null for non-vector-hit chunks
            }


    # Sort chunks by source, then index for logical flow
    sorted_chunks = sorted(unique_chunks.values(), key=lambda x: (x["source"], x["index"]))

    # Create the context string
    context = "\n\n---\n\n".join([chunk["text"] for chunk in sorted_chunks])

    # Prepare metadata list (remove temporary large index if it was used)
    metadata = []
    for chunk in sorted_chunks:
        meta_item = chunk.copy()
        if meta_item["index"] == float('inf'):
            meta_item["index"] = None # Or -1, or omit
        meta_item.pop("id", None) # Remove elementId from final metadata if desired
        metadata.append(meta_item)

    return context, metadata


def retrieve_context(tx, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Executes the PathRAG query in Neo4j combining vector search and graph traversal.
    Targets specific entity labels created during enrichment.
    """
    results = []

    # --- 1. Vector Search ---
    # Find initial chunks semantically similar to the query
    vec_search_query = f"""
    CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding) YIELD node, score
    WHERE node.embedding_model = $model_name // Ensure we get BGE chunks
    RETURN elementId(node) AS id, score, node // Return node for property access
    """
    try:
        vector_hits = tx.run(vec_search_query,
                             index_name=BGE_VECTOR_INDEX_NAME,
                             top_k=top_k,
                             query_embedding=query_embedding,
                             model_name=BGE_MODEL_NAME).data()
        initial_chunk_nodes = {hit['id']: hit for hit in vector_hits}
        print(f"DEBUG: Initial vector search found {len(initial_chunk_nodes)} hits.")
        if not initial_chunk_nodes:
            return []
    except Exception as e:
        print(f"ERROR during vector search: {e}")
        return []

    # --- 2. Combined Graph Traversal Query ---
    initial_ids = list(initial_chunk_nodes.keys())
    scores_map = {hit_id: hit['score'] for hit_id, hit in initial_chunk_nodes.items()}

    # Construct the label filter string for the WHERE clause
    # e.g., "WHERE entity:Company OR entity:Quarter OR entity:Person ..."
    entity_label_filter = " OR ".join([f"entity:{label}" for label in ENTITY_LABELS])

    combined_query = f"""
    // Start with the initial vector hits identified by their element IDs
    MATCH (initial_chunk:Chunk) WHERE elementId(initial_chunk) IN $initial_ids

    // Collect initial chunks along with their potential neighbors
    WITH initial_chunk
    OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(initial_chunk)
    OPTIONAL MATCH (initial_chunk)-[:NEXT_CHUNK]->(next:Chunk)

    // Find SPECIFIC entities mentioned in initial chunks
    WITH initial_chunk, prev, next
    OPTIONAL MATCH (initial_chunk)-[:MENTIONS]->(entity)
    WHERE {entity_label_filter} // <-- Filter for nodes with expected entity labels
    WITH initial_chunk, prev, next, collect(DISTINCT entity) AS mentioned_entities

    // Find *other* BGE chunks mentioning the *same* valid entities
    WITH initial_chunk, prev, next, mentioned_entities
    OPTIONAL MATCH (other_chunk:Chunk)-[:MENTIONS]->(entity_match)
    WHERE entity_match IN mentioned_entities // Match only the entities collected above
      AND elementId(other_chunk) <> elementId(initial_chunk) // Exclude the initial hit itself
      AND other_chunk.embedding_model = $model_name // Ensure other chunks are also BGE

    // Collect all distinct relevant chunk nodes found through any path
    WITH initial_chunk, prev, next, other_chunk
    // Create a list of all node variables from the matches
    WITH [initial_chunk, prev, next, other_chunk] AS all_potential_nodes
    UNWIND all_potential_nodes AS node_var // Unwind the list to get individual nodes
    WITH DISTINCT node_var // Deduplicate the nodes
    WHERE node_var IS NOT NULL // Filter out nulls potentially introduced by OPTIONAL MATCH

    // Return properties needed for context formatting
    RETURN
        elementId(node_var) AS chunk_id,
        node_var.text AS text,
        node_var.source AS source,
        node_var.method AS method,
        node_var.chunk_index AS index,
        // Include vector score for initial hits, null otherwise
        CASE WHEN elementId(node_var) IN $initial_ids THEN $scores_map[elementId(node_var)] ELSE null END as score
    """

    try:
        combined_results = tx.run(combined_query,
                                  initial_ids=initial_ids,
                                  scores_map=scores_map,
                                  model_name=BGE_MODEL_NAME).data()
        print(f"DEBUG: Combined query returned {len(combined_results)} potential context chunks.")
        return combined_results
    except Exception as e:
        print(f"ERROR during combined graph traversal query: {e}")
        # Fallback: return just the initial vector hits if traversal fails
        fallback_results = []
        for hit_id, hit_data in initial_chunk_nodes.items():
            node_props = hit_data.get('node', {}).get('properties', {}) # Safely access properties
            fallback_results.append({
                "chunk_id": hit_id,
                "text": node_props.get("text"),
                "source": node_props.get("source"),
                "method": node_props.get("method"),
                "index": node_props.get("chunk_index"),
                "score": hit_data.get("score")
            })
        return fallback_results


def get_pathrag_context(driver: GraphDatabase.driver, user_query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Main function to get PathRAG context for a user query.
    Connects to Neo4j, embeds query, runs retrieval, formats context.
    Returns context string, metadata list, and context token count.
    """
    if not driver:
        print("ERROR: Neo4j driver is not available.")
        return "", [], 0

    print(f"\nRetrieving context for query: '{user_query}'")
    # 1. Embed Query (using function from embedding_processor.py)
    # Using the recommended instruction prefix for BGE retrieval queries
    query_instruction = "Represent this sentence for searching relevant passages: "
    query_embedding = get_embedding(query_instruction + user_query)

    if query_embedding is None:
        print("ERROR: Failed to generate query embedding.")
        return "", [], 0
    print(f"DEBUG: Query embedding generated (dimension: {len(query_embedding)}).")

    # 2. Run Retrieval Query in Transaction
    context_results = []
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            context_results = session.execute_read(retrieve_context, query_embedding, top_k)
    except Exception as e:
        print(f"ERROR: Neo4j session failed during retrieval: {e}")
        return "", [], 0 # Return empty on session error

    if not context_results:
        print("Warning: No context chunks retrieved from graph.")
        return "", [], 0

    # 3. Format Context
    context_string, metadata = format_context(context_results)
    print(f"DEBUG: Formatted context from {len(metadata)} unique chunks.")

    # 4. Count Context Tokens (using function from embedding_processor.py)
    context_tokens = count_tokens(context_string) # Uses BGE tokenizer

    return context_string, metadata, context_tokens

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure driver is created if running standalone
    main_driver = None
    try:
        load_dotenv() # Ensure .env is loaded
        NEO4J_URI_LOCAL = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        NEO4J_USER_LOCAL = os.getenv("NEO4J_USERNAME", "neo4j")
        NEO4J_PASSWORD_LOCAL = os.getenv("NEO4J_PASSWORD", "your_password")

        main_driver = GraphDatabase.driver(NEO4J_URI_LOCAL, auth=(NEO4J_USER_LOCAL, NEO4J_PASSWORD_LOCAL), max_connection_lifetime=3600)
        main_driver.verify_connectivity()
        print("Neo4j connection successful for example.")

        # --- Test Queries ---
        test_queries = [
            "What was Apple's iPhone revenue in Q3 2022?",
            "Compare Microsoft and Amazon cloud services revenue.",
            "Which executives were mentioned in Intel's Q2 report?",
            "Find information about NVIDIA's R&D spending trends."
        ]

        for test_query in test_queries:
            context, meta, tokens = get_pathrag_context(main_driver, test_query, top_k=3) # Use smaller k for testing

            print("\n" + "="*40)
            print(f"QUERY: {test_query}")
            print("="*40)

            print("\n--- Retrieved Context ---")
            print(context if context else ">>> No Context Found <<<")
            print("\n--- Metadata ---")
            if meta:
                for m in meta:
                    print(f"  - Source: {m.get('source')}, Method: {m.get('method')}, Index: {m.get('index')}, Score: {m.get('score', 'N/A')}")
            else:
                print("  (No metadata)")
            print(f"\n--- Context Tokens (BGE Tokenizer): {tokens} ---")
            print("="*40 + "\n")
            time.sleep(1) # Pause slightly between queries

    except Exception as e:
        print(f"ERROR in example usage: {e}")
    finally:
        if main_driver:
            main_driver.close()
            print("Neo4j connection closed.")