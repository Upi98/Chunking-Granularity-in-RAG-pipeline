# src/evaluate_neo4j.py
import os, time, csv, json, datetime, pandas as pd, requests, textwrap
from neo4j import GraphDatabase
from embedding_processor import count_tokens                      # :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
from naive_retriever import naive_context
from pathrag_retriever import get_pathrag_context                 # :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}

### ----- config -----
CSV_IN        = "../evaluation_qa.csv"
CSV_OUT       = "../results/neo4j_eval.csv"
K             = 5                            # identical k for both modes
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "gemma3:4b")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
### ------------------

def llm_answer(question, context):
    prompt = textwrap.dedent(f"""
        Answer using only this context. If unsure, say you don't know.

        Context:
        {context}

        Question: {question}
        Answer:
    """)
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                      timeout=180)
    r.raise_for_status()
    return r.json()["response"].strip()

def load_qa(csv_path):
    return pd.read_csv(csv_path).fillna("").to_dict("records")

def evaluate_mode(label, retr_fn, qa_rows, results):
    print(f"\n=== {label} (k={K}) ===")
    for i, row in enumerate(qa_rows, 1):
        q = row["Question"]; truth = row["Answer"]
        ctx, meta = retr_fn(q)
        ans = llm_answer(q, ctx) if ctx else "No context"
        results.append({
            "mode": label,
            "idx":  i,
            "question": q,
            "answer":  ans,
            "ground_truth": truth,
            "context_tokens": count_tokens(ctx),
            "meta": json.dumps(meta)
        })
        print(f"  {label} Q#{i}/{len(qa_rows)} done")

def main():
    qa_rows = load_qa(CSV_IN)
    results = []
    t0 = time.time()

    # ---- naive ----
    evaluate_mode("naive", lambda q: naive_context(q, K), qa_rows, results)

    # ---- path‑rag ----
    evaluate_mode("pathrag",
                  lambda q: get_pathrag_context(driver, q, top_k=K)[:2],  # get_pathrag_context returns (ctx, meta, tok)
                  qa_rows, results)

    pd.DataFrame(results).to_csv(CSV_OUT, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nDone → {CSV_OUT} ({len(results)} rows)  elapsed {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
