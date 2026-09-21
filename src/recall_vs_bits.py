"""
recall_vs_bits.py — The scientifically fair benchmark: at a FIXED number of
output bits, does FlyHash's sparse-expand-then-WTA code preserve nearest-
neighbor structure better than SimHash's dense random-hyperplane code?

This isolates the quality of the hash FUNCTION itself, independent of any
indexing trickery — it's the comparison used in the original FlyHash paper
(Dasgupta, Stevens & Navlakha, 2017, Science).

All Hamming-distance ranking is vectorized as a single matrix multiply:
for binary codes a, b in {0,1}^d:
    hamming(a, b) = sum(a) + sum(b) - 2 * dot(a, b)
so ranking every query against every stored code is one matmul, not a
python loop — this is what makes the benchmark itself fast regardless of
which hash is being evaluated.
"""

from __future__ import annotations
import time
import numpy as np

from flyhash import FlyHash, SimHash
#from benchmark import load_embeddings
from real_embeddings import load_embeddings


def vectorized_hamming_ranking(codes: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """
    codes:   (n, d) binary matrix of stored codes
    queries: (q, d) binary matrix of query codes
    returns: (q, n) Hamming distance matrix
    """
    codes = codes.astype(np.int16)
    queries = queries.astype(np.int16)
    code_sums = codes.sum(axis=1)          # (n,)
    query_sums = queries.sum(axis=1)       # (q,)
    dots = queries @ codes.T               # (q, n)
    return query_sums[:, None] + code_sums[None, :] - 2 * dots


def recall_at_k_from_ranking(pred_dists, exact_dists, k=10):
    """
    pred_dists, exact_dists: (q, n) distance matrices from the same query
    set to the same corpus. Recall@k = overlap between each method's own
    top-k and the exact (ground truth) top-k, averaged over queries.
    """
    pred_topk = np.argsort(pred_dists, axis=1)[:, :k]
    exact_topk = np.argsort(exact_dists, axis=1)[:, :k]
    recalls = []
    for p, e in zip(pred_topk, exact_topk):
        recalls.append(len(set(p.tolist()) & set(e.tolist())) / k)
    return float(np.mean(recalls))


def run(n_docs=100, embed_dim=128, n_queries=50, k=10,
        bit_budgets=(128, 256, 512, 1024, 2048)):
    texts, embeddings = load_embeddings(n_docs, embed_dim, n_clusters=50)
    # Use the embeddings' ACTUAL dimensionality, not the requested embed_dim —
    # some loaders (e.g. real_embeddings.py's sentence-transformers loader)
    # ignore embed_dim because the model fixes its own output size (384 for
    # MiniLM-L6-v2), so trusting the parameter instead of the real shape
    # causes a matmul dimension mismatch in FlyHash/SimHash construction.
    actual_dim = embeddings.shape[1]

    rng = np.random.default_rng(1)
    query_idx = rng.choice(len(texts), size=n_queries, replace=False)
    queries = embeddings[query_idx]

    # Ground truth: exact L2 nearest neighbors on the original dense embeddings.
    diffs = queries[:, None, :] - embeddings[None, :, :]
    exact_dists = np.linalg.norm(diffs, axis=2)  # (q, n)

    print(f"{'bits':>6} | {'FlyHash recall@'+str(k):>16} | {'SimHash recall@'+str(k):>16} "
          f"| {'Fly build(s)':>12} | {'Sim build(s)':>12}")
    print("-" * 75)

    results = []
    for bits in bit_budgets:
        # FlyHash: expand into `bits` Kenyon cells, ~5% winner-take-all,
        # each cell wired to ~10% of the input dims (matches the fly's ratios).
        t0 = time.perf_counter()
        fly = FlyHash(actual_dim, hash_dim=bits, sparsity=0.10, wta_percent=0.05)
        fly_codes = fly.hash(embeddings)
        fly_queries = fly.hash(queries)
        fly_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        sim = SimHash(actual_dim, hash_dim=bits)
        sim_codes = sim.hash(embeddings)
        sim_queries = sim.hash(queries)
        sim_time = time.perf_counter() - t0

        fly_dists = vectorized_hamming_ranking(fly_codes, fly_queries)
        sim_dists = vectorized_hamming_ranking(sim_codes, sim_queries)

        fly_recall = recall_at_k_from_ranking(fly_dists, exact_dists, k)
        sim_recall = recall_at_k_from_ranking(sim_dists, exact_dists, k)

        print(f"{bits:>6} | {fly_recall:>16.3f} | {sim_recall:>16.3f} "
              f"| {fly_time:>12.3f} | {sim_time:>12.3f}")
        results.append({"bits": bits, "flyhash_recall": fly_recall,
                         "simhash_recall": sim_recall})

    return results


if __name__ == "__main__":
    run()
