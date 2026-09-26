"""
multi_seed_benchmark.py — Phase 2, statistical rigor: reruns the corrected
FlyHash vs SimHash comparison (recall_vs_bits.py's protocol) across
multiple random seeds and reports mean +/- std instead of a single number.

Every number in this project up to this point (including the GloVe result
that resolved the Phase 1 discrepancy) came from ONE random draw of the
projection matrices. FlyHash's sparse projection and SimHash's random
hyperplanes are both randomly initialized, so a single run tells you what
happened with one particular random seed, not what happens on average or
how much the result varies. This script closes that gap.

What varies across seeds: the random projection matrices for both FlyHash
and SimHash (via the `seed` parameter each already accepts). What stays
fixed across seeds: the dataset and the query sample, so the ONLY source
of variation being measured is projection-matrix randomness, isolated
from query-sampling variation. (Varying the query sample too is a
reasonable further extension, not done here to keep this run's variance
attributable to one clear cause.)

Usage:
    python multi_seed_benchmark.py                  # GloVe, 10 seeds
    python multi_seed_benchmark.py --dataset synthetic --seeds 10
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
    elif name == "digits":
        from digits_embeddings import load_embeddings
        texts, embeddings = load_embeddings(n_docs=n_docs)
        return texts, embeddings
    elif name == "real":
        # Real MiniLM sentence embeddings. Needs internet (downloads the
        # model from huggingface.co on first run) — run this on a machine
        # with internet access, not the sandbox. SAMPLE_CORPUS in
        # real_embeddings.py has 100 unique sentences, so n_docs<=100 avoids
        # repeated/duplicate embeddings (see that file's docstring).
        from real_embeddings import load_embeddings
        texts, embeddings = load_embeddings(n_docs=n_docs)
        return texts, embeddings
    elif name == "mnist":
        # Full MNIST (60k images, 784 dims) via sklearn's fetch_openml.
        # Needs internet on first run (downloads + caches via OpenML).
        from mnist_embeddings import load_embeddings
        texts, embeddings = load_embeddings(n_docs=n_docs)
        return texts, embeddings
    else:
        raise ValueError(f"Unknown dataset: {name}")


def run(dataset: str = "glove", n_docs: int = 5000, n_queries: int = 200,
        n_seeds: int = 10, R: int = 100,
        bit_budgets=(16, 32, 64, 128, 256), flyhash_pool_size: int = 2000):

    texts, embeddings = load_dataset(dataset, n_docs)
    actual_dim = embeddings.shape[1]
    print(f"Dataset: {dataset} | {len(texts)} records | dim={actual_dim} | "
          f"{n_seeds} seeds | mAP@{R}\n")

    # Fixed query sample across all seeds, so variation measured is purely
    # from the projection matrices, not from which queries were drawn.
    query_rng = np.random.default_rng(1)
    query_idx = query_rng.choice(len(texts), size=min(n_queries, len(texts)), replace=False)
    queries = embeddings[query_idx]

    diffs = queries[:, None, :] - embeddings[None, :, :]
    exact_dists = np.linalg.norm(diffs, axis=2)

    # results[bits] = {"fly": [map_seed0, map_seed1, ...], "sim": [...]}
    results = {bits: {"fly": [], "sim": []} for bits in bit_budgets}

    for seed in range(n_seeds):
        for bits in bit_budgets:
            wta_percent = bits / flyhash_pool_size
            fly = FlyHash(actual_dim, hash_dim=flyhash_pool_size, sparsity=0.10,
                           wta_percent=wta_percent, seed=seed)
            sim = SimHash(actual_dim, hash_dim=bits, seed=seed)

            fly_codes = fly.hash(embeddings)
            fly_queries = fly.hash(queries)
            sim_codes = sim.hash(embeddings)
            sim_queries = sim.hash(queries)

            fly_dists = vectorized_hamming_ranking(fly_codes, fly_queries)
            sim_dists = vectorized_hamming_ranking(sim_codes, sim_queries)

            fly_map = compute_map_at_r(fly_dists, exact_dists, R)
            sim_map = compute_map_at_r(sim_dists, exact_dists, R)

            results[bits]["fly"].append(fly_map)
            results[bits]["sim"].append(sim_map)

        print(f"  seed {seed+1}/{n_seeds} done")

    print(f"\n{'k (bits)':>9} | {'FlyHash mAP@'+str(R)+' (mean +/- std)':>28} | "
          f"{'SimHash mAP@'+str(R)+' (mean +/- std)':>28} | {'gap > 2*std?':>12}")
    print("-" * 90)

    summary = []
    for bits in bit_budgets:
        fly_vals = np.array(results[bits]["fly"])
        sim_vals = np.array(results[bits]["sim"])
        fly_mean, fly_std = fly_vals.mean(), fly_vals.std()
        sim_mean, sim_std = sim_vals.mean(), sim_vals.std()
        gap = fly_mean - sim_mean
        combined_std = np.sqrt(fly_std**2 + sim_std**2)
        robust = "YES" if gap > 2 * combined_std else "no"

        print(f"{bits:>9} | {fly_mean:>16.3f} +/- {fly_std:<7.3f} | "
              f"{sim_mean:>16.3f} +/- {sim_std:<7.3f} | {robust:>12}")

        summary.append({
            "bits": bits, "fly_mean": fly_mean, "fly_std": fly_std,
            "sim_mean": sim_mean, "sim_std": sim_std, "gap": gap, "robust": robust,
        })

    print(f"\n'gap > 2*std?' checks whether the FlyHash-SimHash gap exceeds twice")
    print(f"the combined standard deviation — a simple, conservative check that")
    print(f"the difference is unlikely to be explained by projection-matrix noise.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="glove",
                         choices=["glove", "synthetic", "digits", "real", "mnist"])
    parser.add_argument("--n_docs", type=int, default=5000)
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--R", type=int, default=100)
    args = parser.parse_args()

    run(dataset=args.dataset, n_docs=args.n_docs, n_queries=args.n_queries,
        n_seeds=args.seeds, R=args.R)
