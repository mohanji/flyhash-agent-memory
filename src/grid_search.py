"""
grid_search.py — Phase 2c: hyperparameter ablation. Checks whether the
FlyHash-over-SimHash result depends on the two hyperparameters that were
otherwise left at fixed defaults throughout this project: `sparsity` (what
fraction of input dims each Kenyon cell samples from) and `flyhash_pool_size`
(m, the total number of Kenyon cells winners are chosen from).

Why this matters for the write-up: every result so far used sparsity=0.10
and pool_size=2000 without ever checking whether the result is sensitive to
those choices. A reviewer's natural question is "did you just get lucky with
your hyperparameters?" This answers that directly, on the same GloVe dataset
already used throughout, with 3 seeds per cell (fewer than the 10-seed main
result, since this sweeps many more configurations — the point here is
whether the qualitative pattern holds, not a fresh publication-grade number
for each cell).

Usage:
    python grid_search.py --dataset glove --n_docs 5000 --seeds 3
    python grid_search.py --dataset synthetic --seeds 3
"""

from __future__ import annotations
import argparse
import numpy as np

from flyhash import FlyHash, SimHash
from recall_vs_bits import vectorized_hamming_ranking
from map_metric import compute_map_at_r


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


def run(dataset: str = "glove", n_docs: int = 5000, n_queries: int = 200,
        n_seeds: int = 3, R: int = 100, bits: int = 64,
        sparsity_grid=(0.05, 0.10, 0.20),
        pool_size_grid=(500, 1000, 2000, 4000)):
    """
    Fixes the bit budget `bits` (the number of FlyHash winning bits / SimHash
    output bits) and sweeps sparsity x pool_size, reporting FlyHash's mAP@R
    (mean +/- std over n_seeds) at each combination, alongside the fixed
    SimHash baseline at the same bit budget for reference.
    """
    texts, embeddings = load_dataset(dataset, n_docs)
    actual_dim = embeddings.shape[1]
    print(f"Dataset: {dataset} | {len(texts)} records | dim={actual_dim} | "
          f"k={bits} bits | {n_seeds} seeds | mAP@{R}\n")

    query_rng = np.random.default_rng(1)
    query_idx = query_rng.choice(len(texts), size=min(n_queries, len(texts)), replace=False)
    queries = embeddings[query_idx]

    diffs = queries[:, None, :] - embeddings[None, :, :]
    exact_dists = np.linalg.norm(diffs, axis=2)

    # SimHash baseline at this bit budget, for reference (doesn't depend on
    # FlyHash's hyperparameters, so compute it once).
    sim_vals = []
    for seed in range(n_seeds):
        sim = SimHash(actual_dim, hash_dim=bits, seed=seed)
        sim_codes = sim.hash(embeddings)
        sim_queries = sim.hash(queries)
        sim_dists = vectorized_hamming_ranking(sim_codes, sim_queries)
        sim_vals.append(compute_map_at_r(sim_dists, exact_dists, R))
    sim_mean, sim_std = np.mean(sim_vals), np.std(sim_vals)
    print(f"SimHash baseline @ k={bits}: {sim_mean:.3f} +/- {sim_std:.3f}\n")

    print(f"{'sparsity':>9} | " + " | ".join(f"m={m:>5}" for m in pool_size_grid))
    print("-" * (12 + 14 * len(pool_size_grid)))

    results = {}
    for sparsity in sparsity_grid:
        row_cells = []
        for pool_size in pool_size_grid:
            if bits > pool_size:
                row_cells.append("   n/a (k>m)  ")
                continue
            wta_percent = bits / pool_size
            fly_vals = []
            for seed in range(n_seeds):
                fly = FlyHash(actual_dim, hash_dim=pool_size, sparsity=sparsity,
                               wta_percent=wta_percent, seed=seed)
                fly_codes = fly.hash(embeddings)
                fly_queries = fly.hash(queries)
                fly_dists = vectorized_hamming_ranking(fly_codes, fly_queries)
                fly_vals.append(compute_map_at_r(fly_dists, exact_dists, R))
            mean, std = np.mean(fly_vals), np.std(fly_vals)
            results[(sparsity, pool_size)] = (mean, std)
            row_cells.append(f"{mean:.3f}±{std:.3f}")
        print(f"{sparsity:>9} | " + " | ".join(f"{c:>12}" for c in row_cells))

    print(f"\nEach cell is FlyHash mAP@{R} (mean +/- std over {n_seeds} seeds) at "
          f"k={bits} winning bits, for that (sparsity, pool_size) pair.")
    print(f"For reference, SimHash at the same k={bits}: {sim_mean:.3f} +/- {sim_std:.3f}")
    print(f"Cells below the SimHash baseline would mean that hyperparameter "
          f"choice makes FlyHash worse than the conventional baseline; cells "
          f"at or above it are all 'FlyHash still wins' configurations.")

    return results, (sim_mean, sim_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="glove", choices=["glove", "synthetic"])
    parser.add_argument("--n_docs", type=int, default=5000)
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--bits", type=int, default=64)
    args = parser.parse_args()

    run(dataset=args.dataset, n_docs=args.n_docs, n_queries=args.n_queries,
        n_seeds=args.seeds, R=args.R, bits=args.bits)
