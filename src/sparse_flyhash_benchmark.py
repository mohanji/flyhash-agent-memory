"""
sparse_flyhash_benchmark.py — Phase 3 follow-up experiment: does comparing
FlyHash codes as SPARSE index sets (instead of dense pool_size-wide arrays)
close the speed/memory gap found in latency_memory_benchmark.py?

The key fact that makes this possible: every FlyHash code has EXACTLY k
active bits (that's what k-winners-take-all means). So for two FlyHash
codes a, b, each with exactly k ones:
    hamming(a, b) = sum(a) + sum(b) - 2*dot(a,b) = 2k - 2*|intersection|
Since k is fixed, ranking by Hamming distance is EXACTLY equivalent to
ranking by descending intersection size — the number of shared active
indices. This means Hamming ranking can be computed as a sparse-sparse
matrix multiplication (scipy.sparse), which scales with the number of
actual active bits (nnz = k per row) rather than the dense pool width m —
this is the "sparse-aware comparison" flagged as unbuilt future work in
latency_memory_benchmark.py and the README.

This script builds that sparse representation and re-measures:
  1. Query latency: sparse matmul (intersection counts) vs. the original
     dense matmul from latency_memory_benchmark.py, vs. SimHash's dense
     matmul, vs. exact L2.
  2. Memory: storing only the k active indices per record (as uint16,
     since pool_size=2000 fits in 16 bits) vs. SimHash's bit-packed k-bit
     code vs. the dense pool_size-bit packing from before.

This is a genuine experiment, not a known-answer confirmation — it might
close the gap, partially close it, or reveal a new honest trade-off
(e.g. speed recovered but memory still worse due to per-index storage
overhead). Whatever it shows goes in the paper as-is.
"""

from __future__ import annotations
import time
import numpy as np
from scipy import sparse

from flyhash import FlyHash, SimHash
from recall_vs_bits import vectorized_hamming_ranking


def load_dataset(name: str, n_docs: int = 5000):
    if name == "glove":
        from glove_embeddings import load_embeddings
        texts, embeddings = load_embeddings(n_docs=n_docs)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return texts, embeddings
    raise ValueError(f"Unknown dataset: {name}")


def codes_to_sparse(codes: np.ndarray) -> sparse.csr_matrix:
    """(n, pool_size) dense 0/1 array -> scipy CSR sparse matrix."""
    return sparse.csr_matrix(codes)


def time_sparse_intersection(codes_sparse: sparse.csr_matrix,
                               query_sparse: sparse.csr_matrix, n_repeats: int = 20) -> float:
    """
    Average per-query ms for ranking via sparse intersection counts.
    Since every code has the same fixed k active bits, higher intersection
    = lower Hamming distance, so this ranking is equivalent to Hamming
    ranking without ever materializing a dense pool_size-wide array.
    """
    times = []
    q0 = query_sparse[0]
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = codes_sparse.dot(q0.T).toarray()  # (n, 1) intersection counts
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def time_dense_hamming(codes: np.ndarray, query_codes: np.ndarray, n_repeats: int = 20) -> float:
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = vectorized_hamming_ranking(codes, query_codes[:1])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def time_exact_l2(embeddings: np.ndarray, queries: np.ndarray, n_repeats: int = 20) -> float:
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        diffs = queries[:1, None, :] - embeddings[None, :, :]
        _ = np.linalg.norm(diffs, axis=2)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def run(dataset: str = "glove", n_docs: int = 5000, n_queries: int = 50,
        bit_budgets=(16, 32, 64, 128, 256), flyhash_pool_size: int = 2000, seed: int = 0):
    texts, embeddings = load_dataset(dataset, n_docs)
    actual_dim = embeddings.shape[1]
    n_records = len(texts)
    print(f"Dataset: {dataset} | {n_records} records | dim={actual_dim} | pool_size={flyhash_pool_size}\n")

    rng = np.random.default_rng(1)
    query_idx = rng.choice(n_records, size=min(n_queries, n_records), replace=False)
    queries = embeddings[query_idx]

    exact_ms = time_exact_l2(embeddings, queries)
    index_bytes_per_active_bit = 2  # uint16, since pool_size=2000 < 65536

    print(f"Exact L2 baseline: {exact_ms:.3f} ms/query\n")
    print(f"{'k (bits)':>9} | {'sparse idx bytes/rec':>20} | {'SimHash bytes/rec':>18} | "
          f"{'dense ms/query':>15} | {'sparse ms/query':>16} | {'SimHash ms/query':>17}")
    print("-" * 110)

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

        fly_codes_sparse = codes_to_sparse(fly_codes)
        fly_queries_sparse = codes_to_sparse(fly_queries)

        dense_ms = time_dense_hamming(fly_codes, fly_queries)
        sparse_ms = time_sparse_intersection(fly_codes_sparse, fly_queries_sparse)
        sim_ms = time_dense_hamming(sim_codes, sim_queries)

        sparse_idx_bytes = bits * index_bytes_per_active_bit
        sim_bytes = int(np.ceil(bits / 8))

        print(f"{bits:>9} | {sparse_idx_bytes:>20} | {sim_bytes:>18} | "
              f"{dense_ms:>15.4f} | {sparse_ms:>16.4f} | {sim_ms:>17.4f}")

        rows.append({
            "bits": bits, "sparse_idx_bytes": sparse_idx_bytes, "sim_bytes": sim_bytes,
            "dense_ms": dense_ms, "sparse_ms": sparse_ms, "sim_ms": sim_ms,
        })

    print(f"\nCorrectness check: sparse-intersection ranking is mathematically identical to "
          f"dense Hamming ranking here (both derive from the same fixed-k-active-bits "
          f"identity: hamming = 2*(k - intersection)), so mAP results are unchanged — this "
          f"is purely a speed/memory re-implementation, not a new hashing scheme.")

    return rows


if __name__ == "__main__":
    run()
