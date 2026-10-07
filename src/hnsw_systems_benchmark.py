"""
hnsw_systems_benchmark.py — supervisor feedback item #6: "add a practical
dense-vector/HNSW baseline as a SEPARATE systems comparison."

This is explicitly NOT part of the matched-active-bit protocol used
everywhere else in this project (Section 3.2 of the paper). HNSW
(Malkov & Yashunin, 2016/2020) is a graph index over full-precision
vectors; FlyHash/SimHash are compact hash codes compared by Hamming
distance. They are different points in the retrieval design space, and
this script does not pretend otherwise — it reports mAP@R, memory, and
query latency for all of: exact L2 (ground truth), FAISS HNSW (the
dominant practical production ANN method), FlyHash-sparse (this
project's best resource-aware variant, from sparse_flyhash_benchmark.py),
and SimHash, side by side, so a reader asking "would I actually use this
instead of normal vector retrieval?" has a direct, honestly-labeled
answer.

Usage:
    python hnsw_systems_benchmark.py --dataset glove
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import faiss
from scipy import sparse

from flyhash import FlyHash, SimHash
from map_metric import compute_map_at_r
from multi_seed_benchmark import load_dataset


def time_exact_l2(embeddings: np.ndarray, queries: np.ndarray, n_repeats: int = 20):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        diffs = queries[:1, None, :] - embeddings[None, :, :]
        _ = np.linalg.norm(diffs, axis=2)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def build_hnsw_index(embeddings: np.ndarray, M: int = 32, ef_construction: int = 200):
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings.astype(np.float32))
    return index


def hnsw_map_at_r(index, queries: np.ndarray, exact_dists: np.ndarray, R: int, ef_search: int):
    index.hnsw.efSearch = ef_search
    n_corpus = exact_dists.shape[1]
    distances, indices = index.search(queries.astype(np.float32), R)
    # Build a (n_queries, n_corpus) surrogate distance matrix: returned
    # candidates get their real distance; everything else gets +inf, so
    # compute_map_at_r correctly treats "not returned in top-R" the same
    # way it treats "ranked beyond R" for the hash methods.
    pred_dists = np.full((queries.shape[0], n_corpus), np.inf, dtype=np.float32)
    for qi in range(queries.shape[0]):
        valid_idx = indices[qi][indices[qi] >= 0]
        valid_dist = distances[qi][: len(valid_idx)]
        pred_dists[qi, valid_idx] = valid_dist
    return compute_map_at_r(pred_dists, exact_dists, R)


def hnsw_latency(index, queries: np.ndarray, R: int, ef_search: int, n_repeats: int = 20):
    index.hnsw.efSearch = ef_search
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        index.search(queries[:1].astype(np.float32), R)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def hnsw_index_bytes_per_record(index, n_records: int) -> float:
    buf = faiss.serialize_index(index)
    return len(buf) / n_records


def run(dataset: str = "glove", n_docs: int = 5000, n_queries: int = 50, R: int = 100,
        bit_budgets=(16, 32, 64, 128, 256), flyhash_pool_size: int = 2000, seed: int = 0,
        hnsw_M: int = 32, hnsw_ef_construction: int = 200, hnsw_ef_search_values=(16, 64, 128)):

    texts, embeddings = load_dataset(dataset, n_docs)
    actual_dim = embeddings.shape[1]
    n_records = len(texts)
    print(f"Dataset: {dataset} | {n_records} records | dim={actual_dim} | "
          f"mAP@{R}, {n_queries} queries\n")

    rng = np.random.default_rng(1)
    query_idx = rng.choice(n_records, size=min(n_queries, n_records), replace=False)
    queries = embeddings[query_idx]

    diffs = queries[:, None, :] - embeddings[None, :, :]
    exact_dists = np.linalg.norm(diffs, axis=2)
    exact_ms = time_exact_l2(embeddings, queries)
    exact_bytes = actual_dim * 4  # float32

    print(f"Exact L2 baseline: {exact_ms:.3f} ms/query, {exact_bytes} bytes/record\n")

    print("--- FAISS HNSW (M={}, efConstruction={}) ---".format(hnsw_M, hnsw_ef_construction))
    index = build_hnsw_index(embeddings, M=hnsw_M, ef_construction=hnsw_ef_construction)
    hnsw_bytes = hnsw_index_bytes_per_record(index, n_records)
    hnsw_rows = []
    print(f"{'efSearch':>9} | {'mAP@' + str(R):>10} | {'ms/query':>10} | {'bytes/rec':>10}")
    for ef in hnsw_ef_search_values:
        m = hnsw_map_at_r(index, queries, exact_dists, R, ef_search=ef)
        lat = hnsw_latency(index, queries, R, ef_search=ef)
        print(f"{ef:>9} | {m:>10.4f} | {lat:>10.4f} | {hnsw_bytes:>10.1f}")
        hnsw_rows.append({"efSearch": ef, "map": m, "ms_per_query": lat, "bytes_per_record": hnsw_bytes})

    print("\n--- FlyHash (sparse-set comparison) vs. SimHash, matched active-bit k ---")
    print(f"{'k':>5} | {'FlyHash mAP':>11} | {'FlyHash ms/q':>12} | {'FlyHash B/rec':>13} | "
          f"{'SimHash mAP':>11} | {'SimHash ms/q':>12} | {'SimHash B/rec':>13}")
    hash_rows = []
    for bits in bit_budgets:
        wta_percent = bits / flyhash_pool_size
        fly = FlyHash(actual_dim, hash_dim=flyhash_pool_size, sparsity=0.10,
                      wta_percent=wta_percent, seed=seed)
        sim = SimHash(actual_dim, hash_dim=bits, seed=seed)

        fly_codes = fly.hash(embeddings)
        fly_queries = fly.hash(queries)
        sim_codes = sim.hash(embeddings)
        sim_queries = sim.hash(queries)

        # Sparse-set comparison (the resource-aware variant from
        # sparse_flyhash_benchmark.py), since that is FlyHash's best
        # resource profile and the fairer practical comparison point.
        fly_codes_sparse = sparse.csr_matrix(fly_codes)
        fly_queries_sparse = sparse.csr_matrix(fly_queries)

        def fly_dists_for_map():
            inter = fly_codes_sparse.dot(fly_queries_sparse.T).toarray().T.astype(np.int32)
            return 2 * bits - 2 * inter

        fly_dists = fly_dists_for_map()
        sim_dists_arr = ((sim_codes[None, :, :] != sim_queries[:, None, :]).sum(axis=2)).astype(np.float32)

        fly_map = compute_map_at_r(fly_dists, exact_dists, R)
        sim_map = compute_map_at_r(sim_dists_arr, exact_dists, R)

        n_rep = 20
        t0 = time.perf_counter()
        for _ in range(n_rep):
            _ = fly_codes_sparse.dot(fly_queries_sparse[0].T).toarray()
        fly_ms = (time.perf_counter() - t0) / n_rep * 1000

        t0 = time.perf_counter()
        for _ in range(n_rep):
            _ = (sim_codes != sim_queries[0]).sum(axis=1)
        sim_ms = (time.perf_counter() - t0) / n_rep * 1000

        fly_bytes = bits * 2  # uint16 per active index
        sim_bytes = int(np.ceil(bits / 8))

        print(f"{bits:>5} | {fly_map:>11.4f} | {fly_ms:>12.4f} | {fly_bytes:>13} | "
              f"{sim_map:>11.4f} | {sim_ms:>12.4f} | {sim_bytes:>13}")

        hash_rows.append({
            "bits": bits, "fly_map": fly_map, "fly_ms": fly_ms, "fly_bytes": fly_bytes,
            "sim_map": sim_map, "sim_ms": sim_ms, "sim_bytes": sim_bytes,
        })

    print("\nThis is a SYSTEMS comparison, not a matched-active-bit comparison: HNSW's")
    print("mAP/latency/memory depend on (M, efConstruction, efSearch), not on a 'bits'")
    print("parameter comparable to k. Read across rows within each table, not across")
    print("tables at matched row position.")

    return {"exact_ms": exact_ms, "exact_bytes": exact_bytes, "hnsw": hnsw_rows, "hash": hash_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="glove",
                         choices=["glove", "synthetic", "digits", "real", "mnist"])
    parser.add_argument("--n_docs", type=int, default=5000)
    parser.add_argument("--n_queries", type=int, default=50)
    parser.add_argument("--R", type=int, default=100)
    args = parser.parse_args()
    run(dataset=args.dataset, n_docs=args.n_docs, n_queries=args.n_queries, R=args.R)
