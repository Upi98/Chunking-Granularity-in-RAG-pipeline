# src/enrich_graph.py

import os
import re
import time
import spacy
from neo4j import GraphDatabase
from dotenv import load_dotenv

# --- Load Configuration ---
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
NEO4J_DATABASE = "neo4j"

# Load spaCy NLP model
print("Loading spaCy model...")
NLP = spacy.load("en_core_web_sm")
print("spaCy model loaded.")

# --- Constants and Patterns ---
KNOWN_COMPANIES = {"INTC", "MSFT", "NVDA", "AAPL", "AMZN", "Intel", "Microsoft", "NVIDIA", "Apple", "Amazon"}
QUARTER_PATTERN = re.compile(r"(Q[1-4])[\s-]*(\d{4})")

SEGMENT_KEYWORDS = {"iPhone", "Mac", "Azure", "AWS", "Data Center", "Windows", "Surface", "IoT", "Cloud", "Advertising"}
METRIC_KEYWORDS = {"Gross Margin", "Net Income", "Operating Income", "Revenue", "Operating Expenses", "Cash Flow", "R&D"}

SEGMENT_ALIASES = {
    "Amazon Web Services": "AWS",
    "Azure Services": "Azure",
    "Data Centre": "Data Center",
    "Cloud Services": "Cloud",
    "iPhone 14": "iPhone",
}

# --- Neo4j Driver Setup ---
driver = None
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), max_connection_lifetime=3600)
    driver.verify_connectivity()
    print("Neo4j connection successful.")
except Exception as e:
    print(f"ERROR: Neo4j connection failed: {e}")
    driver = None

def run_query(tx, query, parameters=None):
    result = tx.run(query, parameters or {})
    return [r for r in result]

def create_constraints(tx):
    print("Ensuring constraints exist...")
    queries = [
        "CREATE CONSTRAINT unique_company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE;",
        "CREATE CONSTRAINT unique_quarter IF NOT EXISTS FOR (q:Quarter) REQUIRE (q.year, q.quarter_id) IS UNIQUE;",
        "CREATE CONSTRAINT unique_segment IF NOT EXISTS FOR (s:Segment) REQUIRE s.name IS UNIQUE;",
        "CREATE CONSTRAINT unique_metric IF NOT EXISTS FOR (m:Metric) REQUIRE m.name IS UNIQUE;"
    ]
    for query in queries:
        try:
            tx.run(query)
            print(f"Constraint created/verified for: {query.split(' FOR ')[1].split(') ')[0]})")
        except Exception as e:
            print(f"Error creating constraint: {e}")
    print("Constraints setup complete.")

def link_sequential_chunks(tx):
    query = """
        MATCH (c1:Chunk), (c2:Chunk)
        WHERE c1.source = c2.source
          AND c1.chunk_index IS NOT NULL AND c2.chunk_index IS NOT NULL
          AND c1.chunk_index + 1 = c2.chunk_index
          AND c1.embedding_model = 'bge-base-en-v1.5'
          AND c2.embedding_model = 'bge-base-en-v1.5'
        WITH c1, c2
        MERGE (c1)-[r:NEXT_CHUNK]->(c2)
        RETURN count(r) as relationships_created
    """
    result = tx.run(query).single()
    print(f"Created/Merged {result['relationships_created']} :NEXT_CHUNK relationships.")

def process_chunk_batch_for_entities(tx, chunk_batch):
    entities_to_merge = {
        "Company": {}, "Quarter": {}, "Segment": {}, "Metric": {}, "RiskFactor": {}
    }
    mentions_found = {}

    for chunk_info in chunk_batch:
        chunk_id = chunk_info['id']
        text = chunk_info['text']
        mentions_found[chunk_id] = []

        # Known companies
        for company in KNOWN_COMPANIES:
            if company.lower() in text.lower():
                entity_props = {"name": company}
                entities_to_merge["Company"][company] = entity_props
                mentions_found[chunk_id].append({"label": "Company", "props": entity_props})

        # Quarter pattern
        for match in QUARTER_PATTERN.finditer(text):
            q_id = match.group(1)
            year = int(match.group(2))
            key = f"{year}-{q_id}"
            entity_props = {"year": year, "quarter_id": q_id}
            entities_to_merge["Quarter"][key] = entity_props
            mentions_found[chunk_id].append({"label": "Quarter", "props": entity_props})

        # NLP-based filtering
        doc = NLP(text)
        for ent in doc.ents:
            ent_text = ent.text.strip()

            # Segment recognition
            if ent_text in SEGMENT_KEYWORDS or ent_text in SEGMENT_ALIASES:
                canonical = SEGMENT_ALIASES.get(ent_text, ent_text)
                entity_props = {"name": canonical}
                entities_to_merge["Segment"][canonical] = entity_props
                mentions_found[chunk_id].append({"label": "Segment", "props": entity_props})

            # Metric keywords in context
            elif any(metric.lower() in text.lower() for metric in METRIC_KEYWORDS):
                metric = ent_text
                entity_props = {"name": metric}
                entities_to_merge["Metric"][metric] = entity_props
                mentions_found[chunk_id].append({"label": "Metric", "props": entity_props})

    # MERGE nodes
    cypher_statements = []
    for label, nodes in entities_to_merge.items():
        if not nodes:
            continue
        merge_query = ""
        if label == "Quarter":
            merge_query = f"UNWIND $params_list AS item MERGE (e:{label} {{year: item.year, quarter_id: item.quarter_id}})"
        else:
            merge_query = f"UNWIND $params_list AS item MERGE (e:{label} {{name: item.name}})"
        cypher_statements.append({"query": merge_query, "params": {"params_list": list(nodes.values())}})

    # MERGE relationships
    mentions_by_label = {}
    for chunk_id, ents in mentions_found.items():
        for ent in ents:
            mentions_by_label.setdefault(ent["label"], []).append({"chunk_id": chunk_id, "entity_props": ent["props"]})

    for label, mentions in mentions_by_label.items():
        if not mentions:
            continue
        match_props = "{year: mention.entity_props.year, quarter_id: mention.entity_props.quarter_id}" if label == "Quarter" else "{name: mention.entity_props.name}"
        rel_query = f"""
            UNWIND $mentions_param AS mention
            MATCH (c) WHERE elementId(c) = mention.chunk_id
            MATCH (e:{label} {match_props})
            MERGE (c)-[:MENTIONS]->(e)
        """
        cypher_statements.append({"query": rel_query, "params": {"mentions_param": mentions}})

    for stmt in cypher_statements:
        tx.run(stmt["query"], **stmt["params"])
    return {"status": "Processed Batch"}

# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    if not driver:
        print("Neo4j connection error. Exiting.")
    else:
        try:
            with driver.session(database=NEO4J_DATABASE) as session:
                session.execute_write(create_constraints)
                session.execute_write(link_sequential_chunks)

            batch_size = 10
            skip_count = 0
            processed = 0

            while True:
                print(f"\nFetching batch (SKIP {skip_count}, LIMIT {batch_size})...")
                with driver.session(database=NEO4J_DATABASE, default_access_mode="READ") as session:
                    query = """
                        MATCH (c:Chunk)
                        WHERE c.embedding_model = 'bge-base-en-v1.5'
                        WITH c ORDER BY elementId(c)
                        RETURN elementId(c) as id, c.text as text
                        SKIP $skip_param LIMIT $limit_param
                    """
                    chunk_batch = session.execute_read(run_query, query, parameters={
                        "skip_param": skip_count,
                        "limit_param": batch_size
                    })

                if not chunk_batch:
                    print("No more chunks to process.")
                    break

                with driver.session(database=NEO4J_DATABASE) as session:
                    session.execute_write(process_chunk_batch_for_entities, chunk_batch)

                skip_count += len(chunk_batch)
                processed += len(chunk_batch)
                print(f"Processed {processed} chunks so far.")

        finally:
            driver.close()
            print("Neo4j connection closed.")
