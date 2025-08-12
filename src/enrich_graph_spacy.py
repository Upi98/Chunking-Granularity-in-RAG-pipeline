# src/enrich_graph_spacy_optimized.py

import os
import re
import time
import logging
import spacy # Import spaCy
from spacy.tokens import Doc # Import Doc type hint
from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError
from dotenv import load_dotenv
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Set # Added Set

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Configuration ---
load_dotenv(override=True)
NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: Optional[str] = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
EMBEDDING_MODEL_FILTER: Optional[str] = os.getenv("EMBEDDING_MODEL_FILTER")

# --- spaCy Model Loading ---
SPACY_MODEL_NAME = "en_core_web_lg" # Using large model for better accuracy
NLP = None
logging.info(f"Loading spaCy model: {SPACY_MODEL_NAME}...")
try:
    NLP = spacy.load(SPACY_MODEL_NAME)
    logging.info("spaCy model loaded successfully.")
except OSError:
    logging.error(f"Could not download/load spaCy model '{SPACY_MODEL_NAME}'.")
    logging.info(f"Try running: python -m spacy download {SPACY_MODEL_NAME}")
    exit(1) # Exit if essential NLP model fails


# --- Constants and Patterns (Optimized Focus) ---
# Core companies - rely on spaCy's ORG for others unless specified
KNOWN_COMPANIES: Set[str] = {"INTC", "MSFT", "NVDA", "AAPL", "AMZN",
                             "Intel", "Microsoft", "NVIDIA", "Apple", "Amazon"
                             } 

QUARTER_PATTERN: re.Pattern = re.compile(r"\b(Q[1-4])[\s-]*(\d{4})\b") # Use word boundaries

# Core financial metrics - use context check for MONEY entities
METRIC_KEYWORDS: Set[str] = {
    "revenue", "net sales", "sales", "gross margin", "net income", "income", "profit", "loss",
    "operating income", "operating expenses", "expenses", "r&d", "research and development",
    "selling, general and administrative", "sg&a", "cash flow", "tax rate", "earnings per share", "eps",
    "inventory", "debt", "assets", "liabilities", "equity", "liquidity", "stock", "shares",
    "buyback", "repurchase", "dividends"
}
# Currencies often indicate financial context
CURRENCY_SYMBOLS: Set[str] = {"$", "€", "£", "¥"}
CURRENCY_CODES: Set[str] = {"USD", "EUR", "GBP", "JPY"}

# Core product segments/brands
SEGMENT_KEYWORDS: Set[str] = {
    "iPhone", "Mac", "iPad", "Watch", "Services", "App Store", # Apple
    "Azure", "Windows", "Surface", "Xbox", "LinkedIn", "Server", "Microsoft Cloud", # Microsoft
    "AWS", "Amazon Web Services", "Online stores", "Physical stores", "Third-party seller services", "Subscription services", "Advertising services", # Amazon
    "Data Center", "Gaming", "Professional Visualization", "Automotive", # NVIDIA
    "Client Computing Group", "CCG", "Data Center and AI", "DCAI", "Network and Edge", "NEX", "Mobileye", "Intel Foundry Services", "IFS", # Intel
    "Cloud" # Generic
}
SEGMENT_ALIASES: Dict[str, str] = {
    "Amazon Web Services": "AWS", "Azure Services": "Azure",
    "Data Centre": "Data Center", "Cloud Services": "Cloud",
    "iPhone 14": "iPhone", "iPhone 15": "iPhone", # Example normalization
    "Windows 11": "Windows", "Office 365": "Office Commercial",
    # Add other relevant aliases from your data
}


# --- Neo4j Driver Setup ---
# (Same as before)
driver: Optional[Driver] = None
if not NEO4J_PASSWORD or not NEO4J_URI:
    logging.critical("Neo4j URI or password not found. Set NEO4J_URI and NEO4J_PASSWORD. Exiting.")
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
    logging.critical(f"Neo4j connection failed: Server not available at {NEO4J_URI}. Details: {e}. Exiting.")
    driver = None
except Exception as e:
    logging.critical(f"ERROR: Neo4j connection failed: {e}", exc_info=True)
    driver = None

# --- Neo4j Helper Functions ---
# (Same as before)
def run_query(tx: Transaction, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    logging.debug(f"Executing Cypher: {query[:100]}... with params: {parameters is not None}")
    result = tx.run(query, parameters or {})
    return [r.data() for r in result]

def create_constraints(tx: Transaction) -> None:
    logging.info("Ensuring constraints exist...")
    # Adjusted labels based on optimized focus
    constraints_to_ensure = {
        "Chunk": "id", "Company": "name", "Location": "name",
        "Metric": "name", "Segment": "name", "Quarter": None, # Handled separately
   
    }
    created_count = 0

    # Composite key for Quarter
    quarter_constraint_name = "unique_Quarter_year_quarter_id" # Renamed to match convention
    check_query_quarter = "SHOW CONSTRAINTS YIELD name WHERE name = $constraint_name RETURN count(*) as count"
    result_q = tx.run(check_query_quarter, constraint_name=quarter_constraint_name).single()
    if result_q and result_q['count'] == 0:
        try:
            # Use IS NODE KEY for composite constraints
            tx.run(f"CREATE CONSTRAINT {quarter_constraint_name} IF NOT EXISTS FOR (q:Quarter) REQUIRE (q.year, q.quarter_id) IS NODE KEY;")
            logging.info(f"Created constraint: {quarter_constraint_name}")
            created_count +=1
        except Exception as e:
            logging.error(f"Error creating constraint {quarter_constraint_name}: {e}")
    else:
         logging.debug(f"Constraint {quarter_constraint_name} already exists or error checking.")

    # Standard unique constraints
    for label, prop in constraints_to_ensure.items():
        if prop is None: continue # Skip Quarter here
        constraint_name = f"unique_{label}_{prop}"
        check_query = "SHOW CONSTRAINTS YIELD name WHERE name = $constraint_name RETURN count(*) as count"
        result = tx.run(check_query, constraint_name=constraint_name).single()
        if result and result['count'] == 0:
            try:
                tx.run(f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE;")
                logging.info(f"Created constraint: {constraint_name}")
                created_count +=1
            except Exception as e:
                 logging.error(f"Error creating constraint {constraint_name}: {e}")
        else:
            logging.debug(f"Constraint {constraint_name} already exists or error checking.")

    logging.info(f"Constraints setup check complete. {created_count} constraints newly created.")

def link_sequential_chunks(tx: Transaction) -> None:
    # (Same as before)
    logging.info("Linking sequential chunks...")
    where_clause = f"WHERE c1.chunk_index IS NOT NULL AND c1.embedding_model = '{EMBEDDING_MODEL_FILTER}'" if EMBEDDING_MODEL_FILTER else "WHERE c1.chunk_index IS NOT NULL"
    query = f"""
        MATCH (d:Document)-[:CONTAINS]->(c1:Chunk)
        {where_clause}
        WITH d, c1 ORDER BY c1.chunk_index ASC
        WITH d, collect(c1) as chunks_in_doc
        UNWIND range(0, size(chunks_in_doc)-2) as i
        WITH chunks_in_doc[i] as c1, chunks_in_doc[i+1] as c2
        WHERE c1.chunk_index + 1 = c2.chunk_index // Ensure they are truly sequential
        MERGE (c1)-[r:NEXT_CHUNK]->(c2)
        RETURN count(r) as relationships_created
    """
    try:
        result = tx.run(query).single()
        relationships_created = result['relationships_created'] if result else 0
        logging.info(f"Created/Merged {relationships_created} :NEXT_CHUNK relationships.")
    except Exception as e:
         logging.error(f"Error linking sequential chunks: {e}", exc_info=True)


# --- Optimized spaCy Data Extraction ---
def extract_optimized_spacy_data(chunk_batch: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """ Extracts entities relevant to financial Q&A using spaCy. """
    if not NLP:
        logging.error("spaCy model not loaded. Cannot extract entities.")
        return defaultdict(lambda: defaultdict(dict)), defaultdict(list)

    entities_to_merge = defaultdict(lambda: defaultdict(dict)) # {label: {key: props}}
    mentions_to_create = defaultdict(list) # {chunk_element_id: [{"label": L, "key": K, "props": P}]}
    # Relationship extraction is omitted in this optimized version for simplicity

    # SpaCy labels to check and map
    spacy_labels_to_neo4j = {
        "ORG": "Company",
        "PRODUCT": "Segment", # Map PRODUCT to Segment, requires checking keywords
        "GPE": "Location",    # Geopolitical Entity (Countries, States, Cities)
        "LOC": "Location",    # Non-GPE locations (less common, might be less relevant)
        "MONEY": "Metric",    # Requires context check for metric keywords
        "DATE": "Quarter"     # Requires parsing for Qx YYYY format
        # Add LAW if needed for legal questions
    }

    for chunk_info in chunk_batch:
        chunk_element_id = chunk_info.get('id')
        text = chunk_info.get('text')

        if not chunk_element_id or not text or text.isspace():
            logging.debug(f"Skipping chunk with missing ID or text: {chunk_info}")
            continue

        logging.debug(f"Optimized spaCy processing for chunk (elementId: {chunk_element_id})...")
        doc: Doc = NLP(text)

        processed_entity_keys_in_chunk = set() # Track keys (name or quarter key) processed for this chunk

        # --- Extraction Logic ---

        # 1. Regex for Quarters (applied to full text first)
        for match in QUARTER_PATTERN.finditer(text):
            q_id = match.group(1).upper()
            year_str = match.group(2)
            try:
                year = int(year_str)
                key = f"{year}-{q_id}"
                if key not in processed_entity_keys_in_chunk:
                    props = {"year": year, "quarter_id": q_id, "name": key}
                    entities_to_merge["Quarter"][key] = props
                    mentions_to_create[chunk_element_id].append({"label": "Quarter", "key": key, "props": props})
                    processed_entity_keys_in_chunk.add(key)
            except ValueError:
                logging.warning(f"Invalid year '{year_str}' found by QUARTER_PATTERN in chunk {chunk_element_id}")

        # 2. spaCy NER iteration with prioritized checks
        for ent in doc.ents:
            ent_text = ent.text.strip()
            spacy_label = ent.label_

            if not ent_text or ent_text.lower() in {"q1", "q2", "q3", "q4"}: # Skip empty, already processed, or trivial date parts
                continue

            neo4j_label: Optional[str] = None
            key = ent_text.lower() # Use lower case for key matching generally
            props = {"name": ent_text} # Store original casing in properties

            # --- Prioritized Checks ---
            # a) Known Company? (Check original text against set)
            if spacy_label == "ORG" or ent_text in KNOWN_COMPANIES:
                 # Prioritize KNOWN_COMPANIES list if text matches
                 matched_known_comp = next((comp for comp in KNOWN_COMPANIES if comp.lower() == ent_text.lower()), None)
                 if matched_known_comp:
                      key = matched_known_comp # Use canonical name from set as key
                      props["name"] = matched_known_comp
                 else:
                      # If ORG but not in known list, use the found text
                      key = ent_text
                      props["name"] = ent_text
                 neo4j_label = "Company"

            # b) Known Segment/Product? (Check canonical name against keywords)
            elif spacy_label == "PRODUCT" or ent_text in SEGMENT_KEYWORDS or ent_text in SEGMENT_ALIASES:
                canonical_segment = SEGMENT_ALIASES.get(ent_text, ent_text)
                if canonical_segment in SEGMENT_KEYWORDS:
                    key = canonical_segment
                    props["name"] = canonical_segment
                    neo4j_label = "Segment"


            # c) Potential Metric? (Check MONEY label and context)
            elif spacy_label == "MONEY":
                # Check context (sentence) for metric keywords or currency symbols/codes
                sentence_text_lower = ent.sent.text.lower()
                is_metric_related = False
                potential_metric_name = ent_text # Default if no specific keyword found

                for keyword in METRIC_KEYWORDS:
                    if keyword in sentence_text_lower:
                        is_metric_related = True
                        # Try to find a more specific metric name near the money entity
                        # This logic can be complex; starting simple: use keyword if close?
                        # For now, we'll use the keyword *if* found, otherwise the original money text
                        # This is heuristic and might need refinement
                        # A better approach might involve dependency parsing or more advanced NLP
                        potential_metric_name = keyword.capitalize() # Use the keyword found nearby
                        break # Take first keyword match

                if not is_metric_related: # Also check for currency symbols as indicator
                    if any(sym in ent.sent.text for sym in CURRENCY_SYMBOLS) or \
                       any(code in ent.sent.text for code in CURRENCY_CODES):
                        is_metric_related = True
                        potential_metric_name = f"Financial Value ({ent_text})" # Generic metric name

                if is_metric_related:
                    # Use the identified metric term (or a generic one) as the node name/key
                    key = potential_metric_name
                    props["name"] = potential_metric_name
                    props["original_value_text"] = ent_text # Keep original monetary value text if needed
                    neo4j_label = "Metric"

            # e) Potential Quarter/Date? (Check DATE label, parse if possible)
            elif spacy_label == "DATE":
                 q_match = QUARTER_PATTERN.search(ent_text) # Search within the entity text
                 if q_match:
                     q_id = q_match.group(1).upper()
                     year_str = q_match.group(2)
                     try:
                         year = int(year_str)
                         key = f"{year}-{q_id}"
                         props = {"year": year, "quarter_id": q_id, "name": key}
                         neo4j_label = "Quarter"
                     except ValueError:
                         logging.debug(f"Could not parse year from DATE entity '{ent_text}'")
                         neo4j_label = None # Don't create node if parsing fails

            # f) General Location (GPE)? (Less focus based on Q&A, but can keep)
            elif spacy_label == "GPE":
                 key = ent_text
                 props["name"] = ent_text
                 neo4j_label = "Location"

            # --- Add Entity and Mention if a label was assigned and not already processed ---
            if neo4j_label and key not in processed_entity_keys_in_chunk:
                entities_to_merge[neo4j_label][key] = props
                mentions_to_create[chunk_element_id].append({"label": neo4j_label, "key": key, "props": props})
                processed_entity_keys_in_chunk.add(key)

    # Return extracted entities and mentions (no relationships in this version)
    return entities_to_merge, mentions_to_create


# --- Neo4j Writing Function ---
# Reusing the relationship writing logic from previous script, but only MENTIONS will be populated
# If you add relationship extraction logic above, this function should handle it via APOC.
def write_spacy_extraction_results(tx: Transaction,
                                   entities_to_merge: Dict[str, Dict[str, Dict[str, Any]]],
                                   mentions_to_create: Dict[str, List[Dict[str, Any]]]) -> Tuple[int, int]:
    """ Writes spaCy-extracted entities and their mentions to Neo4j using APOC. """
    nodes_created = 0
    mentions_rels_created = 0
    start_time = time.time()

    # 1. MERGE Entity Nodes using apoc.merge.node
    logging.debug("Starting entity node merge using apoc.merge.node...")
    entity_merge_params = []
    node_merge_count = 0
    for label, nodes in entities_to_merge.items():
        node_merge_count += len(nodes)
        for key, props in nodes.items():
             # Determine identifying properties based on label
             if label == "Quarter":
                 ident_props = {"year": props.get("year"), "quarter_id": props.get("quarter_id")}
             else: # Assume name is key
                 ident_props = {"name": props.get("name")}
             if all(v is not None for v in ident_props.values()):
                entity_merge_params.append({
                    "label": label, "ident_props": ident_props, "props_on_create": props
                })
             else:
                 logging.warning(f"Skipping entity merge due to missing identifier: Label={label}, Props={props}")

    nodes_processed_count = 0
    if entity_merge_params:
        node_merge_query = """
        UNWIND $params AS p
        CALL apoc.merge.node([p.label], p.ident_props, p.props_on_create, {}) YIELD node
        RETURN count(node) as nodes_processed_count
        """
        try:
            result = tx.run(node_merge_query, params=entity_merge_params).single()
            nodes_processed_count = result['nodes_processed_count'] if result else 0
        except Exception as e:
            logging.error(f"Error merging nodes using apoc.merge.node: {e}. Ensure APOC is installed.")
            raise # Re-raise to abort transaction if node merge fails

    logging.debug(f"Finished processing {nodes_processed_count} entity node merges in {time.time()-start_time:.2f}s")
    node_write_time = time.time()

    # 2. MERGE MENTIONS Relationships (Chunk -> Entity)
    if mentions_to_create:
        logging.debug("Starting MENTIONS relationship merge...")
        mention_params_list = []
        mention_count = 0
        for chunk_id, mentions in mentions_to_create.items():
            mention_count += len(mentions)
            for mention in mentions:
                entity_props = mention.get("props", {})
                entity_label = mention.get("label")
                if not entity_label: continue

                # Ensure props needed for matching exist
                if entity_label == 'Quarter' and ('year' not in entity_props or 'quarter_id' not in entity_props):
                    logging.warning(f"Skipping Quarter mention merge due to missing props: {mention}")
                    continue
                elif entity_label != 'Quarter' and 'name' not in entity_props:
                    logging.warning(f"Skipping {entity_label} mention merge due to missing name: {mention}")
                    continue

                mention_params_list.append({
                    "chunk_id": chunk_id,
                    "entity_label": entity_label,
                    "entity_props": entity_props
                })

        if mention_params_list:
            rel_query = """
               UNWIND $mentions AS mention
               MATCH (c:Chunk) WHERE elementId(c) = mention.chunk_id

               // Match the entity node - expecting it to exist due to previous step
               CALL apoc.cypher.doIt('MATCH (e:`' + mention.entity_label + '` WHERE ' +
                    CASE mention.entity_label
                       WHEN "Quarter" THEN 'e.year = $props.year AND e.quarter_id = $props.quarter_id'
                       ELSE 'e.name = $props.name'
                    END +
                    ') RETURN e', {props: mention.entity_props, mention:mention}
               ) YIELD value
               WITH c, value.e AS e WHERE e IS NOT NULL

               // Merge the relationship
               MERGE (c)-[r:MENTIONS]->(e)
               ON CREATE SET r.created_at = timestamp()
               RETURN count(r) AS count
               """
               # Using doIt as merge.node inside might create duplicates if constraints failed
               # This assumes nodes were successfully created in step 1
            try:
                result = tx.run(rel_query, mentions=mention_params_list).consume()
                mentions_rels_created = result.counters.relationships_created
            except Exception as e:
                logging.error(f"Error merging MENTIONS relationships: {e}. Ensure APOC plugin is installed.", exc_info=True)
                raise # Re-raise to abort transaction

        logging.debug(f"Finished merging {mention_count} MENTIONS relationships ({mentions_rels_created} created) in {time.time() - node_write_time:.2f}s")

    # Relationship extraction between entities was omitted, so entity_rels_created is 0
    entity_rels_created = 0

    logging.info(f"Batch write summary: Processed node merges, {mentions_rels_created} MENTIONS rels created/merged.")
    return 0, mentions_rels_created # Return rel counts


# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    if not driver:
        logging.critical("Neo4j driver not initialized. Cannot proceed. Exiting.")
        exit(1)
    if not NLP:
         logging.critical("spaCy model failed to load. Cannot proceed with enrichment. Exiting.")
         exit(1)

    logging.info(f"Starting enrichment process using spaCy model: {SPACY_MODEL_NAME}")
    logging.info(f"Targeting Neo4j DB: '{NEO4J_DATABASE}' at {NEO4J_URI}")
    if EMBEDDING_MODEL_FILTER:
        logging.info(f"Filtering Chunks with embedding_model property = '{EMBEDDING_MODEL_FILTER}'")

    overall_start_time = time.time()
    total_processed = 0
    total_rels_created = 0 # Only tracking relationships created

    try:
        # --- Initial Setup ---
        with driver.session(database=NEO4J_DATABASE) as session:
            logging.info(f"Running initial setup on database '{NEO4J_DATABASE}'...")
            session.execute_write(create_constraints)
            session.execute_write(link_sequential_chunks)
            logging.info("Initial setup complete.")

        # --- Batch Processing ---
        batch_size = 500 # Can use larger batches with spaCy
        skip_count = 0

        while True:
            logging.info(f"Fetching chunk batch (SKIP {skip_count}, LIMIT {batch_size})...")
            chunk_batch_data: List[Dict[str, Any]] = []
            fetch_success = False
            try:
                with driver.session(database=NEO4J_DATABASE, default_access_mode="READ") as read_session:
                    # Construct WHERE clause dynamically
                    where_clauses = []
                    params = {"skip_param": skip_count, "limit_param": batch_size}
                    if EMBEDDING_MODEL_FILTER:
                        where_clauses.append("c.embedding_model = $embedding_model")
                        params["embedding_model"] = EMBEDDING_MODEL_FILTER
                    # Add optional WHERE clause to skip already processed chunks if using a flag
                    # where_clauses.append("NOT (c)-[:MENTIONS]->()") # Example: skip if already has mentions

                    where_cypher = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                    query = f"""
                        MATCH (c:Chunk)
                        {where_cypher}
                        WITH c ORDER BY elementId(c)
                        RETURN elementId(c) as id, c.text as text
                        SKIP $skip_param LIMIT $limit_param
                    """
                    chunk_batch_data = read_session.read_transaction(run_query, query, parameters=params)
                    fetch_success = True
            except Neo4jError as e:
                 logging.error(f"Neo4j error fetching chunk batch (skip={skip_count}): {e}. Check query/connection.")
                 break
            except Exception as e:
                 logging.error(f"Unexpected error fetching chunk batch (skip={skip_count}): {e}", exc_info=True)
                 time.sleep(5)
                 break

            if not chunk_batch_data and fetch_success:
                logging.info("No more chunks matching criteria found to process.")
                break
            elif not fetch_success:
                 logging.error("Exiting due to fetch error.")
                 break

            batch_start_time = time.time()
            logging.info(f"Processing batch of {len(chunk_batch_data)} chunks using spaCy...")

            # ----> EXTRACTION STEP <----
            entities_to_merge, mentions_to_create = extract_optimized_spacy_data(chunk_batch_data)

            # ----> WRITING STEP <----
            if entities_to_merge or mentions_to_create:
                write_success = False
                rels_created_in_batch = 0
                try:
                    with driver.session(database=NEO4J_DATABASE) as write_session:
                        # Call the writing function (pass empty list for entity relationships)
                        _, rels_created_in_batch = write_session.execute_write(
                            write_spacy_extraction_results,
                            entities_to_merge,
                            mentions_to_create
                        )
                        total_rels_created += rels_created_in_batch
                        write_success = True
                except Neo4jError as e:
                     logging.error(f"Neo4j error writing spaCy batch results (skip={skip_count}): {e}. Check Cypher/APOC/Data.", exc_info=True)
                except Exception as e:
                    logging.error(f"Unexpected error writing spaCy batch results (skip={skip_count}) to Neo4j: {e}", exc_info=True)

                if not write_success:
                    logging.warning(f"Skipping batch starting at {skip_count} due to write error.")
                    skip_count += len(chunk_batch_data)
                    time.sleep(1)
                    continue
            else:
                 logging.info(f"Nothing extracted by spaCy for this batch (skip={skip_count}).")

            # ----> Update Counts and Log <----
            batch_duration = time.time() - batch_start_time
            total_processed += len(chunk_batch_data)
            skip_count += len(chunk_batch_data) # Increment for the next batch fetch
            logging.info(f"spaCy Batch processed in {batch_duration:.2f}s. Total chunks processed: {total_processed}.")
            logging.info(f"Total MENTIONS relationships created/merged so far: {total_rels_created}")

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
        logging.info(f"Final counts - Chunks processed: {total_processed}, MENTIONS relationships created/merged: {total_rels_created}")