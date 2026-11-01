"""
offline_demo.py -- runs the FlyHash agent-memory demo with NO API key and NO network.

Loads the released benchmark, stores every memory in AgentMemory (FlyHash-backed),
and answers a few questions, printing the recalled memories next to the labeled
ground truth. Uses TF-IDF+SVD embeddings (the same offline embedding as the paper's
benchmark) instead of a downloaded sentence-transformer, so it runs anywhere.
For the MiniLM-backed AgentMemory wrapper see agent_memory.py; for the interactive
Claude-powered chat agent see agent_demo.py.

Usage (from the repo root):
    python src/offline_demo.py
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from memory_store import FlyMemoryStore  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    d = os.path.join(HERE, "..", "datasets", "agent_memory_v1")
    memories = load_jsonl(os.path.join(d, "memories.jsonl"))
    queries = load_jsonl(os.path.join(d, "queries.jsonl"))

    texts = [m["text"] for m in memories]
    tfidf = TfidfVectorizer().fit(texts)
    svd = TruncatedSVD(n_components=256, random_state=0).fit(tfidf.transform(texts))
    embed = lambda xs: svd.transform(tfidf.transform(xs)).astype("float32")

    store = FlyMemoryStore(input_dim=256, hash_dim=2048, sparsity=0.10,
                           wta_percent=0.05, mode="scan")
    store.add_batch(texts, embed(texts))
    print(f"\nStored {len(texts)} memories.\n")

    hits = 0
    for q in queries[:5]:
        results = store.query(embed([q["text"]])[0], k=4)
        truth = {memories[i]["text"] for i in q["relevant_memory_ids"]}
        got = [t for t, _ in results]
        hits += sum(t in truth for t in got)
        print(f"Q: {q['text']}")
        for t in got:
            print(f"   {'[relevant]' if t in truth else '[other]   '} {t}")
        print()
    print(f"Relevant memories in top-4 across 5 queries: {hits}/20")
