"""
benchmark.py — Recall / latency / memory comparison: FlyHash vs FAISS-flat.

Data: locally-generated clustered embeddings (sklearn.datasets.make_blobs),
the standard way to benchmark ANN/hashing methods (mirrors how SIFT/GIST
ANN benchmarks are structured: many tight clusters in high-dim space,
so "recall" means "did we find points from the same semantic cluster").
This needs no network access or model downloads, so it runs anywhere.

To use this on YOUR real data: replace `load_embeddings()` with a loader
that reads your own texts and calls a real embedding model (e.g.
sentence-transformers, or an API embedding call) — everything downstream
(FlyHash, memory stores, recall/latency/memory measurement) is unchanged.
"""

from __future__ import annotations
import time
import numpy as np
from sklearn.datasets import make_blobs

from memory_store import FlyMemoryStore, FaissMemoryStore


def load_embeddings(n_docs: int = 3000, embed_dim: int = 128, seed: int = 0,
                     n_clusters: int = 40):
    """
    Simulates a realistic agent-memory corpus: many short "memories" that
    cluster into ~n_clusters semantic topics (conversations tend to revisit
    a limited set of topics), embedded in embed_dim-dimensional space.
    """
    print(f"Generating {n_docs} synthetic embeddings "
          f"({n_clusters} semantic clusters, dim={embed_dim})...")
    embeddings, cluster_ids = make_blobs(
        n_samples=n_docs, n_features=embed_dim, centers=n_clusters,
        cluster_std=1.8, random_state=seed,
    )
    embeddings = embeddings.astype(np.float32)
    # L2-normalize, since real sentence embeddings are typically normalized.
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    texts = [f"memory_{i:05d} (topic {cluster_ids[i]})" for i in range(n_docs)]
    return texts, embeddings


def recall_at_k(fly_results, exact_results, k):
    """Fraction of exact top-k neighbors also found in fly-hash top-k results."""
    exact_set = {t for t, _ in exact_results[:k]}
    fly_set = {t for t, _ in fly_results[:k]}
    if not exact_set:
        return 0.0
    return len(exact_set & fly_set) / len(exact_set)


def run_benchmark(n_docs=20000, embed_dim=128, n_queries=200, k=10,
                   hash_dim=4000, sparsity=0.10, wta_percent=0.02,
                   n_clusters=60):
    texts, embeddings = load_embeddings(n_docs, embed_dim, n_clusters=n_clusters)

    print(f"\nBuilding stores: {len(texts)} records, embed_dim={embed_dim}, "
          f"hash_dim={hash_dim}, sparsity={sparsity}, wta={wta_percent} "
          f"(k={int(wta_percent*hash_dim)} active cells/record)")

    fly_bucket = FlyMemoryStore(embed_dim, hash_dim, sparsity, wta_percent, mode="bucket")
    fly_scan = FlyMemoryStore(embed_dim, hash_dim, sparsity, wta_percent, mode="scan")
    fly_scan.hasher = fly_bucket.hasher  # share the exact same projection/hash
    exact_store = FaissMemoryStore(embed_dim)

    t0 = time.perf_counter()
    fly_bucket.add_batch(texts, embeddings)
    fly_build_time = time.perf_counter() - t0
    # reuse the same computed codes for the scan-mode store (avoid re-hashing)
    fly_scan.texts = list(fly_bucket.texts)
    fly_scan._active_indices = list(fly_bucket._active_indices)

    t0 = time.perf_counter()
    exact_store.add_batch(texts, embeddings)
    exact_build_time = time.perf_counter() - t0

    # Use a random sample of the corpus itself as queries (simulates "recall
    # a similar past memory" — a realistic agent-memory access pattern).
    rng = np.random.default_rng(1)
    query_idx = rng.choice(len(texts), size=min(n_queries, len(texts)), replace=False)

    recalls_bucket, recalls_scan = [], []
    bucket_times, scan_times, exact_times = [], [], []
    candidates_scanned = []

    for i in query_idx:
        q_emb = embeddings[i]

        t0 = time.perf_counter()
        bucket_res = fly_bucket.query(q_emb, k=k)
        bucket_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        scan_res = fly_scan.query(q_emb, k=k)
        scan_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        exact_res = exact_store.query(q_emb, k=k)
        exact_times.append(time.perf_counter() - t0)

        recalls_bucket.append(recall_at_k(bucket_res, exact_res, k))
        recalls_scan.append(recall_at_k(scan_res, exact_res, k))

        # how many candidates did the bucket approach actually have to rank,
        # vs. the full store size (this is the sublinearity payoff)
        query_active = set(fly_bucket._hash_to_active_indices(
            fly_bucket.hasher.hash(np.asarray(q_emb, dtype=np.float32))).tolist())
        n_candidates = len({rid for cell in query_active
                             for rid in fly_bucket._buckets.get(cell, ())})
        candidates_scanned.append(n_candidates)

    fly_mem = fly_bucket.memory_bytes()
    exact_mem = exact_store.memory_bytes()
    avg_candidates = float(np.mean(candidates_scanned))

    print("\n" + "=" * 66)
    print("RESULTS")
    print("=" * 66)
    print(f"Records indexed:                 {len(texts)}")
    print(f"Recall@{k}  bucket mode (vs exact):   {np.mean(recalls_bucket):.3f}")
    print(f"Recall@{k}  scan mode   (vs exact):   {np.mean(recalls_scan):.3f}")
    print(f"Avg candidates ranked, bucket:    {avg_candidates:.0f} "
          f"({100*avg_candidates/len(texts):.1f}% of the store)")
    print(f"Avg query latency  bucket mode:   {np.mean(bucket_times)*1e6:.1f} µs")
    print(f"Avg query latency  scan  mode:    {np.mean(scan_times)*1e6:.1f} µs")
    print(f"Avg query latency  FAISS (exact): {np.mean(exact_times)*1e6:.1f} µs")
    print(f"Build time  FlyHash:              {fly_build_time*1000:.1f} ms")
    print(f"Build time  FAISS:                {exact_build_time*1000:.1f} ms")
    print(f"Memory  FlyHash (sparse indices): {fly_mem/1024:.1f} KB "
          f"({fly_mem/len(texts):.1f} bytes/record)")
    print(f"Memory  FAISS (float32 dense):    {exact_mem/1024:.1f} KB "
          f"({exact_mem/len(texts):.1f} bytes/record)")
    print(f"Memory reduction:                {exact_mem/fly_mem:.1f}x smaller")
    print("=" * 66)

    return {
        "recall_bucket": float(np.mean(recalls_bucket)),
        "recall_scan": float(np.mean(recalls_scan)),
        "candidates_scanned_pct": 100 * avg_candidates / len(texts),
        "bucket_latency_us": float(np.mean(bucket_times) * 1e6),
        "scan_latency_us": float(np.mean(scan_times) * 1e6),
        "exact_latency_us": float(np.mean(exact_times) * 1e6),
        "memory_reduction_x": exact_mem / fly_mem,
    }


if __name__ == "__main__":
    run_benchmark()
