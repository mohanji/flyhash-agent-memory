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
from benchmark import load_embeddings


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


def run(n_docs=5000, embed_dim=128, n_queries=200, k=10,
        bit_budgets=(16, 32, 64, 128, 256), flyhash_pool_size=2000):
    """
    CORRECTED comparison protocol (fixed after diffing against the official
    Fly-LSH reference implementation, github.com/dataplayer12/Fly-LSH).

    The original paper's fair comparison holds the number of WINNING bits
    (hash_length, here `bit_budgets`) equal between FlyHash and conventional
    LSH — NOT the total hash dimensionality. FlyHash's `embedding_size`
    (the Kenyon-cell pool winners are chosen from, `flyhash_pool_size`) is a
    SEPARATE, much larger parameter, independent of the bit budget being
    swept. An earlier version of this script swept `hash_dim` directly while
    holding `wta_percent` fixed, which silently varied FlyHash's *effective*
    code length (its number of active bits) far below the number of bits
    SimHash was given at the same nominal "hash_dim" — an apples-to-oranges
    comparison that made FlyHash look uniformly worse than SimHash across
    every dataset tested (synthetic, TF-IDF, real sentence and GloVe
    embeddings) until this was caught and fixed. See README's "Phase 1"
    section for the full account.
    """
    texts, embeddings = load_embeddings(n_docs, embed_dim, n_clusters=50)
    actual_dim = embeddings.shape[1]

    rng = np.random.default_rng(1)
    query_idx = rng.choice(len(texts), size=n_queries, replace=False)
    queries = embeddings[query_idx]

    diffs = queries[:, None, :] - embeddings[None, :, :]
    exact_dists = np.linalg.norm(diffs, axis=2)

    print(f"{'k (bits)':>9} | {'FlyHash recall@'+str(k):>18} | {'SimHash recall@'+str(k):>18} "
          f"| {'Fly build(s)':>12} | {'Sim build(s)':>12}")
    print("-" * 80)

    results = []
    for bits in bit_budgets:
        wta_percent = bits / flyhash_pool_size
        t0 = time.perf_counter()
        fly = FlyHash(actual_dim, hash_dim=flyhash_pool_size, sparsity=0.10,
                       wta_percent=wta_percent)
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

        print(f"{bits:>9} | {fly_recall:>18.3f} | {sim_recall:>18.3f} "
              f"| {fly_time:>12.3f} | {sim_time:>12.3f}")
        results.append({"bits": bits, "flyhash_recall": fly_recall,
                         "simhash_recall": sim_recall})

    return results


if __name__ == "__main__":
    run()
