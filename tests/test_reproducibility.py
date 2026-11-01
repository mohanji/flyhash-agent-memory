import os
import sys
import json

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from agent_memory_dataset import build_dataset
from flyhash import FlyHash, SimHash


def test_dataset_matches_paper_size():
    memories, queries, rel = build_dataset(n_personas=150, seed=0)
    assert len(memories) == 6000
    assert len(queries) == 150
    avg = sum(len(s) for s in rel) / len(rel)
    assert 3.5 <= avg <= 4.0


def test_dataset_deterministic():
    assert build_dataset(seed=0) == build_dataset(seed=0)


def test_released_files_match_generator():
    d = os.path.join(ROOT, "datasets", "agent_memory_v1")
    memories, queries, rel = build_dataset(n_personas=150, seed=0)
    with open(os.path.join(d, "memories.jsonl"), encoding="utf-8") as f:
        got = [json.loads(l)["text"] for l in f]
    assert got == memories
    with open(os.path.join(d, "queries.jsonl"), encoding="utf-8") as f:
        qs = [json.loads(l) for l in f]
    assert [q["text"] for q in qs] == queries
    assert [set(q["relevant_memory_ids"]) for q in qs] == rel


def test_flyhash_fixed_sparsity_and_locality():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 64)).astype(np.float32)
    fh = FlyHash(input_dim=64, hash_dim=500, wta_percent=0.05, seed=0)
    codes = np.asarray(fh.hash(x)).astype(int)
    assert (codes.sum(axis=1) == 25).all()           # fixed k active bits
    near = x[0] + 0.01 * rng.standard_normal(64).astype(np.float32)
    c0, cn = codes[0], np.asarray(fh.hash(near)).astype(int)
    assert (c0 & cn).sum() > (c0 & codes[1]).sum()    # similar inputs overlap more
