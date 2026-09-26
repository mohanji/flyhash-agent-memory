"""
latency_memory_benchmark.py — Phase 3 prep: honest post-fix memory and query
latency numbers under the CORRECTED equal-bit-budget protocol.

Why this exists: the only memory/latency numbers previously in this project
(README's "Current results" table, from benchmark.py) predate the Phase 1
fix and used the OLD, incorrect comparison design (fixed hash_dim, fixed
wta_percent) on synthetic data via FAISS as the baseline. Those numbers
shouldn't go in a paper that also reports the corrected mAP results — they'd
be measuring a different, already-superseded configuration. This script
measures memory and latency for the SAME corrected protocol used everywhere
else in Phase 2 (fixed k, large fixed pool m, real GloVe data).

What's measured, per bit budget k:
  - Memory per record: packed sparse binary code (both FlyHash and SimHash,
    same k bits -> same packed size) vs. the original float32 embedding.
  - Query latency: brute-force Hamming-distance ranking against the full
    corpus (vectorized numpy), vs. brute-force exact L2 ranking on the
    original float32 embeddings (the "no hashing at all" baseline).

Honest scope note: this is brute-force ranking for BOTH the hashed and
exact cases — there is no sublinear index here (consistent with this
project's earlier finding that the bucket-mode inverted index isn't
actually sublinear at tested scale). This measures the cost of computing
distances/comparisons over already-hashed vs. already-dense data, not an
end-to-end ANN system with indexing. That's the honest scope: it isolates
what hashing itself buys you (smaller, faster-to-compare codes), not what
a production ANN index on top of it would buy you.
"""

from __future__ import annotations
import argparse
import time
import numpy as np

from flyhash import FlyHash, SimHash
from recall_vs_bits import vectorized_hamming_ranking


def load_dataset(name: str, n_docs: int = 5000):
    if name == "glove":
        from glove_embeddings import load_embeddings
        texts, embeddings = load_embeddings(n_docs=n_docs)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return texts, embeddings
    elif name == "synthetic":
        from benchmark import load_embeddings
        texts, embeddings = load_embeddings(n_docs=n_docs, embed_dim=128, n_clusters=50)
        return texts, embeddings
    else:
        raise ValueError(f"Unknown dataset: {name}")


def time_exact_l2(embeddings: np.ndarray, queries: np.ndarray, n_repeats: int = 20) -> float:
    """Average per-query wall-clock time (ms) for brute-force exact L2 ranking."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        diffs = queries[:1, None, :] - embeddings[None, :, :]
        _ = np.linalg.norm(diffs, axis=2)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def time_hamming(codes: np.ndarray, query_codes: np.ndarray, n_repeats: int = 20) -> float:
    """Average per-query wall-clock time (ms) for vectorized Hamming ranking."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = vectorized_hamming_ranking(codes, query_codes[:1])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def run(dataset: str = "glove", n_docs: int = 5000, n_queries: int = 50,
        bit_budgets=(16, 32, 64, 128, 256), flyhash_pool_size: int = 2000,
        seed: int = 0):
    texts, embeddings = load_dataset(dataset, n_docs)
    actual_dim = embeddings.shape[1]
    n_records = len(texts)
    print(f"Dataset: {dataset} | {n_records} records | dim={actual_dim}\n")

    rng = np.random.default_rng(1)
    query_idx = rng.choice(n_records, size=min(n_queries, n_records), replace=False)
    queries = embeddings[query_idx]

    float32_bytes_per_record = actual_dim * 4
    exact_query_ms = time_exact_l2(embeddings, queries)

    print(f"Exact float32 embeddings: {float32_bytes_per_record} bytes/record, "
          f"{exact_query_ms:.3f} ms/query (brute-force L2 over all {n_records} records)\n")

    print(f"{'k (bits)':>9} | {'packed bytes/record':>20} | {'mem reduction':>14} | "
          f"{'FlyHash ms/query':>17} | {'SimHash ms/query':>17} | {'speedup vs exact':>17}")
    print("-" * 105)

    rows = []
    for bits in bit_budgets:
        wta_percent = bits / flyhash_pool_size
        fly = FlyHash(actual_dim, hash_dim=flyhash_pool_size, sparsity=0.10,
                       wta_percent=wta_percent, seed=seed)
        sim = SimHash(actual_dim, hash_dim=bits, seed=seed)

        fly_codes = fly.hash(embeddings)
        fly_queries = fly.hash(queries)
        sim_codes = sim.hash(embeddings)
        sim_queries = sim.hash(queries)

        # Packed size: FlyHash's code lives in a hash_dim=pool_size-wide space
        # but only k bits are ever 1, so pack the FULL pool_size-wide vector
        # (this is the honest packed size — a real implementation could use
        # a sparse index-list representation instead, which would be much
        # smaller still since only k indices need storing; noted as a
        # limitation below rather than silently assumed).
        fly_packed_bytes = int(np.ceil(flyhash_pool_size / 8))
        sim_packed_bytes = int(np.ceil(bits / 8))

        fly_ms = time_hamming(fly_codes, fly_queries)
        sim_ms = time_hamming(sim_codes, sim_queries)

        mem_reduction = float32_bytes_per_record / sim_packed_bytes
        speedup = exact_query_ms / sim_ms if sim_ms > 0 else float("nan")

        print(f"{bits:>9} | {sim_packed_bytes:>20} | {mem_reduction:>13.1f}x | "
              f"{fly_ms:>17.4f} | {sim_ms:>17.4f} | {speedup:>16.1f}x")

        rows.append({
            "bits": bits, "fly_packed_bytes_full_pool": fly_packed_bytes,
            "sim_packed_bytes": sim_packed_bytes, "fly_ms": fly_ms, "sim_ms": sim_ms,
            "mem_reduction_vs_float32": mem_reduction, "speedup_vs_exact": speedup,
        })

    print(f"\nNote on FlyHash memory: FlyHash's code technically lives in a "
          f"{flyhash_pool_size}-bit space (only k bits ever set to 1), so "
          f"packing the full bit-vector costs {int(np.ceil(flyhash_pool_size/8))} "
          f"bytes/record regardless of k — worse than SimHash's k-bit code at the "
          f"SAME k. A sparse index-list representation (storing only the k active "
          f"indices, e.g. as k x 11-bit ints for a 2000-wide pool) would instead cost "
          f"~{bit_budgets[-1] * 2} bytes at k={bit_budgets[-1]} and less at smaller k — "
          f"this project's memory_store.py already uses exactly that sparse-index "
          f"representation for actual storage, so real deployed memory cost is much "
          f"closer to SimHash's than this dense-packing number suggests. This script "
          f"reports the dense-packed number here because it's what makes an apples-to-"
          f"apples Hamming-distance computation via XOR/popcount straightforward; the "
          f"quality (mAP) results elsewhere in this project are unaffected either way, "
          f"since they only depend on which k bits are set, not on the storage format.")
    print(f"\nExact L2 baseline: {exact_query_ms:.3f} ms/query, "
          f"{float32_bytes_per_record} bytes/record.")
    print(f"All timings are brute-force (no ANN index) for both hashed and exact "
          f"cases — see this file's docstring for the honest scope of what that")
    print(f"does and doesn't demonstrate.")

    return rows, exact_query_ms, float32_bytes_per_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="glove", choices=["glove", "synthetic"])
    parser.add_argument("--n_docs", type=int, default=5000)
    parser.add_argument("--n_queries", type=int, default=50)
    args = parser.parse_args()

    run(dataset=args.dataset, n_docs=args.n_docs, n_queries=args.n_queries)
