# src/enrich_graph_mistral.py

import os
import re
import time
import requests
import json
import logging
from neo4j import GraphDatabase, Driver, Session, Transaction # Added specific imports
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError
from dotenv import load_dotenv
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional # Added typing

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Configuration ---
load_dotenv()
NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: Optional[str] = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD") # Read password
NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
EMBEDDING_MODEL_FILTER: Optional[str] = os.getenv("EMBEDDING_MODEL_FILTER")

# --- LLM Configuration ---
LOCAL_LLM_API_URL: Optional[str] = os.getenv("LOCAL_LLM_API_URL")
LOCAL_LLM_MODEL: Optional[str] = os.getenv("LOCAL_LLM_MODEL")
LOCAL_LLM_REQUEST_TIMEOUT: int = int(os.getenv("LOCAL_LLM_REQUEST_TIMEOUT", 180))

# --- LLM Prompt Template ---
# This prompt asks for entities and relationships in a specific JSON format.
# It needs to be tuned based on the specific LLM's ability to follow instructions.
LLM_EXTRACTION_PROMPT: str = """
You are an expert financial analyst assisting with knowledge graph creation by extracting structured information from text snippets of financial reports.

Analyze the following text snippet:
\"\"\"
{chunk_text}
\"\"\"

**Instructions:**

1.  **Identify Entities:** Extract all relevant entities based on the following types:
    * `Company`: Organizations involved in business (e.g., Apple, Intel, subsidiary names).
    * `Person`: Individuals mentioned (e.g., Tim Cook, board members).
    * `Location`: Cities, states, countries relevant to operations or mentioned (e.g., Cupertino, Ireland, USA).
    * `Metric`: Financial terms or key performance indicators (e.g., Revenue, Net Income, Gross Margin, EPS, R&D). Use the exact term found in the text.
    * `Segment`: Business segments or product lines (e.g., iPhone, Mac, AWS, Azure, Data Center, Client Computing). Normalize known aliases (e.g., "Amazon Web Services" -> "AWS", "Data Centre" -> "Data Center").
    * `Quarter`: Financial quarters mentioned (e.g., "Q1 2024", "fourth quarter of 2023"). Extract year and quarter ID (Q1-Q4).

2.  **Identify Relationships:** Extract relationships *between the identified entities* based *only* on information explicitly stated or strongly implied *in the provided text snippet*. Use the following relationship types:
    * `WORKS_FOR` (Person -> Company)
    * `CEO_OF` (Person -> Company)
    * `LOCATED_IN` (Company -> Location)
    * `PARTNERS_WITH` (Company -> Company)
    * `ACQUIRED` (Company -> Company)
    * `REPORTS_METRIC` (Company -> Metric) # If the text associates a company with reporting a metric
    * `OPERATES_SEGMENT` (Company -> Segment) # If the text associates a company with a segment

3.  **Output Format:** Respond *only* with a single, valid JSON object containing two keys: "entities" and "relationships".
    * `entities`: A list of JSON objects, each with `"label"` (string) and `"name"` (string). For `Quarter` entities, also include `"year"` (integer) and `"quarter_id"` (string, e.g., "Q1"). Ensure names are extracted accurately.
    * `relationships`: A list of JSON objects, each with `"type"` (string, e.g., "CEO_OF"), `"from_entity_name"` (string), and `"to_entity_name"` (string). Match names exactly to the extracted entities' names. Ensure relationships only connect entities found in the 'entities' list for this text snippet. If no relationships are found, return an empty list.

**Example JSON Output:**
```json
{{
  "entities": [
    {{"label": "Company", "name": "NVIDIA"}},
    {{"label": "Person", "name": "Jensen Huang"}},
    {{"label": "Segment", "name": "Data Center"}},
    {{"label": "Metric", "name": "Revenue"}},
    {{"label": "Quarter", "name": "Q1 2024", "year": 2024, "quarter_id": "Q1"}}
  ],
  "relationships": [
    {{"type": "CEO_OF", "from_entity_name": "Jensen Huang", "to_entity_name": "NVIDIA"}},
    {{"type": "OPERATES_SEGMENT", "from_entity_name": "NVIDIA", "to_entity_name": "Data Center"}},
    {{"type": "REPORTS_METRIC", "from_entity_name": "NVIDIA", "to_entity_name": "Revenue"}}
  ]
}}
Provide only the JSON object in your response.
"""

--- Neo4j Driver Setup ---
driver: Optional[Driver] = None
if not NEO4J_PASSWORD:
logging.critical("Neo4j password not found in environment variables (NEO4J_PASSWORD). Exiting.")
exit(1)
if not NEO4J_URI:
logging.critical("Neo4j URI not found in environment variables (NEO4J_URI). Exiting.")
exit(1)

try:
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
max_connection_lifetime=7200, connection_timeout=30)
driver.verify_connectivity()
logging.info("Neo4j connection successful.")
except AuthError:
logging.critical(f"Neo4j authentication failed for user '{NEO4J_USER}'. Check credentials. Exiting.")
driver = None
except ServiceUnavailable as e:
logging.critical(f"Neo4j connection failed: Neo4j server not available at {NEO4J_URI}. Details: {e}. Exiting.")
driver = None
except Exception as e:
logging.critical(f"ERROR: Neo4j connection failed: {e}")
driver = None

--- LLM API Function ---
def query_local_llm(prompt_template: str, chunk_text: str) -> Optional[Dict[str, Any]]:
""" Sends text chunk and prompt to the local LLM and parses JSON response. """
if not LOCAL_LLM_API_URL or not LOCAL_LLM_MODEL:
logging.error("Local LLM API URL or Model Name not configured.")
return None

# Basic check for placeholder existence
if "{chunk_text}" not in prompt_template:
     logging.error("Prompt template is missing the required '{chunk_text}' placeholder.")
     return None

full_prompt = prompt_template.format(chunk_text=chunk_text)

# Determine payload structure based on API endpoint type (simple check)
api_endpoint = LOCAL_LLM_API_URL
is_chat_endpoint = "chat" in api_endpoint.split('/')[-1]

if is_chat_endpoint:
    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False,
        "format": "json",
        # "options": { "temperature": 0.2 } # Example option
    }
    logging.debug("Using chat endpoint payload structure.")
else: # Assume generate endpoint
    payload = {
        "model": LOCAL_LLM_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
         # "options": { "temperature": 0.2 } # Example option
    }
    logging.debug("Using generate endpoint payload structure.")

llm_output_str = None # Initialize to ensure it's defined in case of exceptions before assignment
try:
    response = requests.post(
        api_endpoint,
        json=payload,
        timeout=LOCAL_LLM_REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json"} # Ensure header is set
    )
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    response_data = response.json()

    if is_chat_endpoint:
         message = response_data.get("message", {})
         llm_output_str = message.get("content") if isinstance(message, dict) else None
    else: # Assume generate endpoint
         llm_output_str = response_data.get("response")

    if not llm_output_str or not isinstance(llm_output_str, str) or llm_output_str.isspace():
         logging.warning(f"LLM response content is empty or not a string. Raw response: {response_data}")
         return None

    # Clean potential markdown ```json ... ``` markers if format=json didn't strip them
    if llm_output_str.strip().startswith("```json"):
        llm_output_str = llm_output_str.strip()[7:]
        if llm_output_str.strip().endswith("```"):
             llm_output_str = llm_output_str.strip()[:-3]

    parsed_json = json.loads(llm_output_str)
    # Basic validation of expected structure
    if not isinstance(parsed_json, dict) or "entities" not in parsed_json or "relationships" not in parsed_json:
         logging.warning(f"LLM JSON output missing 'entities' or 'relationships' key. Output: {llm_output_str}")
         return None
    return parsed_json

except requests.exceptions.Timeout:
    logging.error(f"Timeout error querying local LLM API ({api_endpoint}) after {LOCAL_LLM_REQUEST_TIMEOUT}s.")
    return None
except requests.exceptions.ConnectionError:
     logging.error(f"Connection error querying local LLM API ({api_endpoint}). Is Ollama running?")
     return None
except requests.exceptions.RequestException as e:
    logging.error(f"Error querying local LLM API ({api_endpoint}): {e}")
    return None
except json.JSONDecodeError as e:
    logging.error(f"Failed to parse JSON response from LLM: {e}")
    logging.error(f"LLM raw response string causing error: '{llm_output_str}'")
    return None
except Exception as e:
    # Catch any other unexpected exceptions
    logging.error(f"An unexpected error occurred during LLM query: {e}", exc_info=True)
    return None
--- Neo4j Helper Functions ---
def run_query(tx: Transaction, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
""" Helper to run query and return list of records """
logging.debug(f"Executing Cypher: {query} with params: {parameters is not None}")
result = tx.run(query, parameters or {})
# Consume the result fully and convert records to dictionaries
return [r.data() for r in result]

def create_constraints(tx: Transaction) -> None:
""" Ensures necessary constraints are created. """
logging.info("Ensuring constraints exist...")
# Unique constraints help prevent duplicate entities and improve performance.
# Adjust labels and properties based on your actual schema.
constraints_to_ensure = {
"Chunk": "id",
"Company": "name",
"Person": "name",
"Location": "name",
"Metric": "name",
"Segment": "name"
# Add other unique constraints as needed
}
# Special case for Quarter (composite key)
quarter_constraint_name = "unique_quarter_year_id"
check_query_quarter = f"SHOW CONSTRAINTS YIELD name WHERE name = $constraint_name RETURN count(*) as count"
result = tx.run(check_query_quarter, constraint_name=quarter_constraint_name).single()
if result and result['count'] == 0:
try:
tx.run(f"CREATE CONSTRAINT {quarter_constraint_name} IF NOT EXISTS FOR (q:Quarter) REQUIRE (q.year, q.quarter_id) IS NODE KEY;")
logging.info(f"Created constraint: {quarter_constraint_name}")
except Exception as e:
logging.error(f"Error creating constraint {quarter_constraint_name}: {e}")
else:
logging.debug(f"Constraint {quarter_constraint_name} already exists or error checking.")

# Standard constraints
for label, prop in constraints_to_ensure.items():
    constraint_name = f"unique_{label}_{prop}"
    check_query = f"SHOW CONSTRAINTS YIELD name WHERE name = $constraint_name RETURN count(*) as count"
    result = tx.run(check_query, constraint_name=constraint_name).single()
    if result and result['count'] == 0:
        try:
            # Use standard CREATE CONSTRAINT syntax
            tx.run(f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE;")
            logging.info(f"Created constraint: {constraint_name}")
        except Exception as e:
             logging.error(f"Error creating constraint {constraint_name}: {e}")
    else:
        logging.debug(f"Constraint {constraint_name} already exists or error checking.")

logging.info("Constraints setup check complete.")
def link_sequential_chunks(tx: Transaction) -> None:
""" Creates :NEXT_CHUNK relationships between sequential chunks in the same document. """
logging.info("Linking sequential chunks...")
# This query assumes chunks have a 'chunk_index' property and are linked to a 'Document'
# It also filters by embedding_model if specified
where_clause = f"WHERE c1.chunk_index IS NOT NULL AND c1.embedding_model = '{EMBEDDING_MODEL_FILTER}'" if EMBEDDING_MODEL_FILTER else "WHERE c1.chunk_index IS NOT NULL"

query = f"""
    MATCH (d:Document)-[:CONTAINS]->(c1:Chunk)
    {where_clause}
    WITH d, c1 ORDER BY c1.chunk_index ASC
    WITH d, collect(c1) as chunks_in_doc
    UNWIND range(0, size(chunks_in_doc)-2) as i
    WITH chunks_in_doc[i] as c1, chunks_in_doc[i+1] as c2
    MERGE (c1)-[r:NEXT_CHUNK]->(c2)
    RETURN count(r) as relationships_created
"""
try:
    result = tx.run(query).single()
    relationships_created = result['relationships_created'] if result else 0
    logging.info(f"Created/Merged {relationships_created} :NEXT_CHUNK relationships.")
except Exception as e:
     logging.error(f"Error linking sequential chunks: {e}", exc_info=True)
--- LLM Data Extraction ---
def extract_llm_data_from_batch(chunk_batch: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
""" Extracts entities and relationships from chunks using the configured local LLM. """
entities_to_merge = defaultdict(lambda: defaultdict(dict)) # {label: {key: props}}
mentions_to_create = defaultdict(list) # {chunk_id: [{"label": L, "key": K, "props": P}]}
relationships_to_create = [] # List of {"type": T, "from_key": K1, "from_label": L1, ...}

for chunk_info in chunk_batch:
    chunk_element_id = chunk_info.get('id') # Neo4j elementId()
    text = chunk_info.get('text')

    if not chunk_element_id or not text or text.isspace():
        logging.debug(f"Skipping chunk with missing ID or text: {chunk_info}")
        continue

    logging.debug(f"Querying LLM for chunk (elementId: {chunk_element_id})...")
    llm_result = query_local_llm(LLM_EXTRACTION_PROMPT, text)

    if llm_result is None:
        logging.warning(f"No valid result from LLM for chunk (elementId: {chunk_element_id}). Skipping.")
        continue

    extracted_entities = llm_result.get("entities", [])
    extracted_relationships = llm_result.get("relationships", [])

    if not isinstance(extracted_entities, list) or not isinstance(extracted_relationships, list):
         logging.warning(f"LLM result for chunk (elementId: {chunk_element_id}) has incorrect format: {llm_result}")
         continue

    # Process extracted entities - build a map for this chunk first
    entity_map_for_chunk = {} # Map name to {"label": L, "key": K, "props": P}
    valid_entities_for_chunk = []
    for entity in extracted_entities:
        label = entity.get("label")
        name = entity.get("name")
        if not label or not isinstance(label, str) or not name or not isinstance(name, str):
            logging.warning(f"Skipping entity with missing/invalid label or name in chunk {chunk_element_id}: {entity}")
            continue

        # Basic cleaning
        name = name.strip()
        label = label.strip().capitalize() # Consistent casing
        if not name: continue # Skip empty names

        props = {"name": name}
        key = name # Default key is the name

        if label == "Quarter":
            year = entity.get("year")
            q_id = entity.get("quarter_id")
            if year is None or q_id is None:
                logging.warning(f"Skipping Quarter entity with missing year/quarter_id in chunk {chunk_element_id}: {entity}")
                continue
            try:
                props["year"] = int(year)
                props["quarter_id"] = str(q_id).strip().upper() # Ensure Q1, Q2 etc.
                if not re.match(r'^Q[1-4]$', props["quarter_id"]):
                     raise ValueError("Invalid quarter format")
                key = f"{props['year']}-{props['quarter_id']}" # Composite key
                # Add original name if different, useful for context
                if entity.get("name") != key:
                     props["original_name"] = entity.get("name")
            except (ValueError, TypeError) as e:
                 logging.warning(f"Skipping Quarter entity with invalid year/quarter_id type/format ('{e}') in chunk {chunk_element_id}: {entity}")
                 continue
        # --- Add normalization/key logic for other labels if needed ---
        # E.g., Company name normalization? Person name splitting?

        # Add to batch merge list (duplicates handled by MERGE)
        entities_to_merge[label][key] = props
        # Add to this chunk's mentions list
        mention_info = {"label": label, "key": key, "props": props}
        mentions_to_create[chunk_element_id].append(mention_info)
        # Add to this chunk's lookup map (using original name from LLM)
        entity_map_for_chunk[name] = mention_info
        valid_entities_for_chunk.append(entity) # Keep track of valid ones

    # Process extracted relationships - only connect entities validly extracted *for this chunk*
    valid_entity_names_for_chunk = {e["name"] for e in valid_entities_for_chunk}

    for rel in extracted_relationships:
        rel_type = rel.get("type")
        from_name = rel.get("from_entity_name")
        to_name = rel.get("to_entity_name")

        # Basic validation
        if not rel_type or not isinstance(rel_type, str) or \
           not from_name or not isinstance(from_name, str) or \
           not to_name or not isinstance(to_name, str):
            logging.warning(f"Skipping relationship with missing/invalid fields in chunk {chunk_element_id}: {rel}")
            continue

        # Ensure both source and target entities were successfully extracted *in this chunk's context*
        # Use the map created *from this chunk's entities*
        if from_name not in entity_map_for_chunk or to_name not in entity_map_for_chunk:
            logging.warning(f"Relationship entities '{from_name}' or '{to_name}' not found in valid entities for chunk {chunk_element_id}. Skipping relationship: {rel}")
            continue

        from_entity_info = entity_map_for_chunk[from_name]
        to_entity_info = entity_map_for_chunk[to_name]

        relationships_to_create.append({
            "type": rel_type.strip().upper().replace(" ", "_"), # Consistent relationship type format
            "from_key": from_entity_info["key"],
            "from_label": from_entity_info["label"],
            "from_props": from_entity_info["props"],
            "to_key": to_entity_info["key"],
            "to_label": to_entity_info["label"],
            "to_props": to_entity_info["props"],
            "metadata": { # Optional: Add context
                "source_chunk_id": chunk_element_id
            }
        })

return entities_to_merge, mentions_to_create, relationships_to_create
--- Neo4j Writing Function ---
def write_llm_extraction_results(tx: Transaction,
entities_to_merge: Dict[str, Dict[str, Dict[str, Any]]],
mentions_to_create: Dict[str, List[Dict[str, Any]]],
relationships_to_create: List[Dict[str, Any]]) -> Tuple[int, int]:
""" Writes LLM-extracted entities, mentions, and relationships to Neo4j using APOC. """
nodes_created = 0
mentions_rels_created = 0
entity_rels_created = 0
start_time = time.time()

# 1. MERGE Entity Nodes using apoc.merge.node for simplicity
# This handles different labels and key properties more dynamically if needed
logging.debug("Starting entity node merge using apoc.merge.node...")
entity_merge_params = []
node_merge_count = 0
for label, nodes in entities_to_merge.items():
    node_merge_count += len(nodes)
    for key, props in nodes.items():
         # Determine identifying properties based on label
         if label == "Quarter":
             ident_props = {"year": props["year"], "quarter_id": props["quarter_id"]}
         else: # Assume name is key for others
             ident_props = {"name": props["name"]}
         entity_merge_params.append({
             "label": label,
             "ident_props": ident_props,
             "props_on_create": props # Set all props if node is created
         })

if entity_merge_params:
    # Use apoc.merge.node - handles creation and ensures unique nodes based on ident_props
    node_merge_query = """
    UNWIND $params AS p
    CALL apoc.merge.node([p.label], p.ident_props, p.props_on_create) YIELD node
    RETURN count(node) as nodes_processed
    """
    # Note: apoc.merge.node doesn't directly return created count in summary
    # We rely on constraints or prior knowledge that MERGE is mostly used
    try:
        tx.run(node_merge_query, params=entity_merge_params).consume()
        # We can't easily get exact created count here without complex checks
        logging.debug(f"Processed {node_merge_count} entity node merges.")
    except Exception as e:
        logging.error(f"Error merging nodes using apoc.merge.node: {e}. Ensure APOC is installed.")
        # Optionally raise e to stop transaction

logging.debug(f"Finished entity node merge attempt in {time.time() - start_time:.2f}s")
node_write_time = time.time()

# 2. MERGE MENTIONS Relationships (Chunk -> Entity)
if mentions_to_create:
    logging.debug("Starting MENTIONS relationship merge...")
    mention_params_list = []
    mention_count = 0
    for chunk_id, mentions in mentions_to_create.items():
        mention_count += len(mentions)
        for mention in mentions:
             mention_params_list.append({
                 "chunk_id": chunk_id, # Neo4j elementId
                 "entity_label": mention["label"],
                 "entity_props": mention["props"] # Pass props needed for matching entity
             })

    # Simplified matching logic using apoc.case
    rel_query = """
       UNWIND $mentions AS mention
       MATCH (c:Chunk) WHERE elementId(c) = mention.chunk_id

       // Match the entity node using MERGE logic (find or create via apoc.merge.node)
       CALL apoc.merge.node([mention.entity_label],
           CASE mention.entity_label WHEN 'Quarter' THEN {year: mention.entity_props.year, quarter_id: mention.entity_props.quarter_id} ELSE {name: mention.entity_props.name} END
       ) YIELD node AS e

       // Merge the relationship
       WITH c, e
       MERGE (c)-[r:MENTIONS]->(e)
       ON CREATE SET r.created_at = timestamp() // Optional: Add timestamp
       RETURN count(r) as count
       """
    try:
        result = tx.run(rel_query, mentions=mention_params_list).consume()
        mentions_rels_created = result.counters.relationships_created
    except Exception as e:
        logging.error(f"Error merging MENTIONS relationships: {e}. Ensure APOC plugin is installed.", exc_info=True)

    logging.debug(f"Finished merging {mention_count} MENTIONS relationships ({mentions_rels_created} created) in {time.time() - node_write_time:.2f}s")
mention_write_time = time.time()


# 3. MERGE Relationships Between Entities
if relationships_to_create:
    logging.debug(f"Starting entity-entity relationship merge for {len(relationships_to_create)} relationships...")
    rel_merge_query = """
       UNWIND $rel_params AS rel_data

       // Find or Create start node (e1) using apoc.merge.node
       CALL apoc.merge.node([rel_data.from_label],
           CASE rel_data.from_label WHEN 'Quarter' THEN {year: rel_data.from_props.year, quarter_id: rel_data.from_props.quarter_id} ELSE {name: rel_data.from_props.name} END, // Identifier
           rel_data.from_props // Properties on create
       ) YIELD node AS e1

       // Find or Create end node (e2) using apoc.merge.node
       CALL apoc.merge.node([rel_data.to_label],
           CASE rel_data.to_label WHEN 'Quarter' THEN {year: rel_data.to_props.year, quarter_id: rel_data.to_props.quarter_id} ELSE {name: rel_data.to_props.name} END, // Identifier
           rel_data.to_props // Properties on create
       ) YIELD node AS e2

       // Merge relationship using dynamic type with apoc.merge.relationship
       CALL apoc.merge.relationship(e1, rel_data.type,
           {}, // Properties to match on relationship (usually none needed for MERGE)
           CASE WHEN rel_data.metadata IS NOT NULL THEN rel_data.metadata ELSE {} END, // Properties to set ON CREATE / ON MATCH
           e2
       ) YIELD rel
       RETURN count(rel) as count
       """
    try:
        result = tx.run(rel_merge_query, rel_params=relationships_to_create).consume()
        entity_rels_created = result.counters.relationships_created
    except Exception as e:
        logging.error(f"Error merging entity relationships using APOC: {e}. Check APOC installation/query.", exc_info=True)

    logging.debug(f"Finished merging entity relationships ({entity_rels_created} created) in {time.time() - mention_write_time:.2f}s")

total_rels = mentions_rels_created + entity_rels_created
total_duration = time.time() - start_time
# Cannot easily get nodes_created accurately from apoc.merge.node summary, log attempt instead
logging.info(f"Batch write summary: Attempted {node_merge_count} node merges, "
             f"{mentions_rels_created} MENTIONS rels created/merged, "
             f"{entity_rels_created} entity-entity rels created/merged in {total_duration:.2f}s.")

# Return created relationship counts, node count is unreliable here
return 0, total_rels
--- MAIN SCRIPT EXECUTION ---
if name == "main":
if not driver:
logging.critical("Neo4j driver not initialized. Cannot proceed. Exiting.")
exit(1)
if not LOCAL_LLM_API_URL or not LOCAL_LLM_MODEL:
logging.critical("Local LLM not configured. Set LOCAL_LLM_API_URL and LOCAL_LLM_MODEL in .env. Exiting.")
exit(1)

logging.info(f"Starting enrichment process using LLM: {LOCAL_LLM_MODEL} via {LOCAL_LLM_API_URL}")
logging.info(f"Targeting Neo4j DB: '{NEO4J_DATABASE}' at {NEO4J_URI}")
if EMBEDDING_MODEL_FILTER:
    logging.info(f"Filtering Chunks with embedding_model property = '{EMBEDDING_MODEL_FILTER}'")

overall_start_time = time.time()
total_processed = 0
total_nodes_created_reported = 0 # Note: Node count from APOC write is inaccurate
total_rels_created = 0

try:
    # --- Initial Setup ---
    with driver.session(database=NEO4J_DATABASE) as session:
        logging.info(f"Running initial setup on database '{NEO4J_DATABASE}'...")
        session.execute_write(create_constraints)
        session.execute_write(link_sequential_chunks)
        logging.info("Initial setup complete.")

    # --- Batch Processing ---
    batch_size = 10 # Start with a small batch size for LLM processing
    skip_count = 0

    while True:
        logging.info(f"Fetching chunk batch (SKIP {skip_count}, LIMIT {batch_size})...")
        chunk_batch_data: List[Dict[str, Any]] = [] # Type hint
        fetch_success = False
        try:
            with driver.session(database=NEO4J_DATABASE, default_access_mode="READ") as read_session:
                # Construct WHERE clause dynamically
                where_clauses = []
                params = {"skip_param": skip_count, "limit_param": batch_size}
                if EMBEDDING_MODEL_FILTER:
                    where_clauses.append("c.embedding_model = $embedding_model")
                    params["embedding_model"] = EMBEDDING_MODEL_FILTER
                # Optional: Add condition to skip already processed chunks if using a flag
                # where_clauses.append("(NOT EXISTS(c.enriched) OR c.enriched = false)")

                where_cypher = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

                query = f"""
                    MATCH (c:Chunk)
                    {where_cypher}
                    WITH c ORDER BY elementId(c) // Use internal ID for stable ordering
                    RETURN elementId(c) as id, c.text as text
                    SKIP $skip_param LIMIT $limit_param
                """
                chunk_batch_data = read_session.read_transaction(run_query, query, parameters=params)
                fetch_success = True
        except Neo4jError as e:
             logging.error(f"Neo4j error fetching chunk batch (skip={skip_count}): {e}. Check query/connection.")
             # Decide if this is recoverable - maybe wait and retry or break
             break # Exit loop on database error
        except Exception as e:
             logging.error(f"Unexpected error fetching chunk batch (skip={skip_count}): {e}", exc_info=True)
             time.sleep(5) # Wait before potentially retrying or breaking
             break # Exit loop on unexpected error

        if not chunk_batch_data and fetch_success:
            logging.info("No more chunks matching criteria found to process.")
            break
        elif not fetch_success:
             logging.error("Exiting due to fetch error.")
             break


        batch_start_time = time.time()
        logging.info(f"Processing batch of {len(chunk_batch_data)} chunks using {LOCAL_LLM_MODEL}...")

        # ----> EXTRACTION STEP <----
        entities_to_merge, mentions_to_create, relationships_to_create = extract_llm_data_from_batch(chunk_batch_data)

        # ----> WRITING STEP <----
        if entities_to_merge or mentions_to_create or relationships_to_create:
            write_success = False
            nodes_created = 0
            rels_created = 0
            try:
                with driver.session(database=NEO4J_DATABASE) as write_session:
                    # Use execute_write for automatic transaction handling (commit/rollback)
                    nodes_created, rels_created = write_session.execute_write(
                        write_llm_extraction_results,
                        entities_to_merge,
                        mentions_to_create,
                        relationships_to_create
                    )
                    total_nodes_created_reported += nodes_created # Keep track (though value might be 0 from APOC)
                    total_rels_created += rels_created
                    write_success = True
            except Neo4jError as e: # Catch specific Neo4j errors during write
                 logging.error(f"Neo4j error writing batch results (skip={skip_count}): {e}. Check Cypher/APOC/Data.", exc_info=True)
                 # Transaction will be rolled back automatically by execute_write
            except Exception as e:
                logging.error(f"Unexpected error writing batch results (skip={skip_count}) to Neo4j: {e}", exc_info=True)

            if not write_success:
                logging.warning(f"Skipping batch starting at {skip_count} due to write error.")
                # Update skip count even if write failed to avoid infinite loop on bad batch
                # But don't count as processed successfully
                skip_count += len(chunk_batch_data)
                time.sleep(2)
                continue # Move to the next batch
        else:
             logging.info(f"Nothing extracted by LLM for this batch (skip={skip_count}).")

        # ----> Update Counts and Log <----
        batch_duration = time.time() - batch_start_time
        total_processed += len(chunk_batch_data)
        skip_count += len(chunk_batch_data) # Increment for the next batch fetch
        logging.info(f"LLM Batch processed in {batch_duration:.2f}s. Total chunks processed: {total_processed}.")
        # Log relationship count as it's more reliable from the summary
        logging.info(f"Total relationships created/merged so far: {total_rels_created}")


except KeyboardInterrupt:
    logging.warning("Process interrupted by user.")
except Exception as e:
    logging.critical(f"An unexpected critical error occurred during main processing loop: {e}", exc_info=True)
finally:
    if driver:
        driver.close()
        logging.info("Neo4j connection closed.")
    overall_end_time = time.time()
    logging.info(f"Enrichment process finished in {overall_end_time - overall_start_time:.2f} seconds.")
    logging.info(f"Final counts - Chunks processed: {total_processed}, Relationships created/merged: {total_rels_created}")