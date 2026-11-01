"""
export_dataset.py -- writes the labeled agent-memory benchmark used in the paper
to plain JSONL files so it can be inspected and reused without running any code.

The dataset is synthetic and fully deterministic (built from templates and a
seeded RNG in agent_memory_dataset.py). No personal data is involved.

Usage (from the repo root):
    python src/export_dataset.py                 # paper configuration
    python src/export_dataset.py --out datasets/custom --n_personas 50 --seed 1

Output files:
    memories.jsonl   {"id": int, "text": str}
    queries.jsonl    {"id": int, "text": str, "relevant_memory_ids": [int, ...]}
    meta.json        generation parameters and summary statistics
"""
from __future__ import annotations
import argparse
import json
import os

from agent_memory_dataset import build_dataset


def export(out_dir: str, n_personas: int, seed: int, n_query_slots_per_persona: int) -> dict:
    memories, queries, relevant_sets = build_dataset(
        n_personas=n_personas, seed=seed,
        n_query_slots_per_persona=n_query_slots_per_persona)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "memories.jsonl"), "w", encoding="utf-8") as f:
        for i, t in enumerate(memories):
            f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for i, (q, rel) in enumerate(zip(queries, relevant_sets)):
            f.write(json.dumps({"id": i, "text": q,
                                "relevant_memory_ids": sorted(rel)},
                               ensure_ascii=False) + "\n")

    meta = {
        "generator": "src/agent_memory_dataset.py::build_dataset",
        "n_personas": n_personas,
        "seed": seed,
        "n_query_slots_per_persona": n_query_slots_per_persona,
        "n_memories": len(memories),
        "n_queries": len(queries),
        "avg_relevant_per_query": round(
            sum(len(s) for s in relevant_sets) / max(1, len(relevant_sets)), 3),
        "license": "MIT (synthetic template-generated text, no personal data)",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="datasets/agent_memory_v1")
    p.add_argument("--n_personas", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_query_slots_per_persona", type=int, default=1)
    a = p.parse_args()
    print(json.dumps(export(a.out, a.n_personas, a.seed, a.n_query_slots_per_persona), indent=2))
