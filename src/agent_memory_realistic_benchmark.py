"""
agent_memory_realistic_benchmark.py — supervisor feedback item #5: evaluate
FlyHash vs. SimHash on the realistically-sized, labeled agent-memory
dataset from agent_memory_dataset.py, using standard mean Average
Precision computed against ACTUAL relevance labels (which memories truly
answer each question, by construction) rather than nearest-neighbor-by-
embedding pseudo-labels.

Embedding: TF-IDF -> truncated SVD, fit on the memory corpus and applied
to both memories and queries (the same offline, network-free approach as
real_text_proxy.py). This is an honest stand-in for a real sentence-
transformer embedding — documented as a limitation in the paper, with a
real-embedding rerun (on a machine with internet access) flagged as a
natural follow-up, the same pattern used for real MiniLM/full MNIST
earlier in this project.

Usage:
    python agent_memory_realistic_benchmark.py --seeds 10
"""

from __future__ import annotations
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from agent_memory_dataset import build_dataset
from flyhash import FlyHash, SimHash
from recall_vs_bits import vectorized_hamming_ranking


def compute_map_labeled(pred_dists: np.ndarray, relevant_sets: list[set]) -> float:
    """
    Standard mean Average Precision over the FULL ranking, using actual
    (non-nearest-neighbor-derived) relevance labels. pred_dists: (n_queries,
    n_corpus). relevant_sets[i]: set of corpus indices relevant to query i.
    """
    n_queries = pred_dists.shape[0]
    ranking = np.argsort(pred_dists, axis=1)  # (n_queries, n_corpus), ascending distance
    aps = np.zeros(n_queries)
    for q in range(n_queries):
        relevant = relevant_sets[q]
        num_relevant = len(relevant)
        if num_relevant == 0:
            continue
        hits = 0
        precision_sum = 0.0
        for rank, idx in enumerate(ranking[q], start=1):
            if idx in relevant:
                hits += 1
                precision_sum += hits / rank
                if hits == num_relevant:
                    break  # all relevant items found; rest of ranking can't add precision
        aps[q] = precision_sum / num_relevant
    return float(np.mean(aps))


def embed(memories: list[str], queries: list[str], embed_dim: int = 256, seed: int = 0):
    tfidf = TfidfVectorizer()
    mem_X = tfidf.fit_transform(memories)
    query_X = tfidf.transform(queries)
    n_components = min(embed_dim, mem_X.shape[1] - 1, mem_X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    mem_emb = svd.fit_transform(mem_X).astype(np.float32)
    query_emb = svd.transform(query_X).astype(np.float32)
    return mem_emb, query_emb


def run(n_personas: int = 150, n_query_slots_per_persona: int = 1,
        embed_dim: int = 256, n_seeds: int = 10,
        bit_budgets=(16, 32, 64, 128, 256), flyhash_pool_size: int = 2000,
        dataset_seed: int = 0):

    memories, queries, relevant_sets = build_dataset(
        n_personas=n_personas,
        n_query_slots_per_persona=n_query_slots_per_persona,
        seed=dataset_seed,
    )
    print(f"Agent-memory benchmark: {len(memories)} memories, {len(queries)} queries, "
          f"avg {sum(len(s) for s in relevant_sets) / len(relevant_sets):.2f} relevant "
          f"memories/query\n")

    mem_emb, query_emb = embed(memories, queries, embed_dim=embed_dim, seed=dataset_seed)
    print(f"Embedding dim after TF-IDF + SVD: {mem_emb.shape[1]}\n")

    results = {bits: {"fly": [], "sim": []} for bits in bit_budgets}

    for seed in range(n_seeds):
        for bits in bit_budgets:
            wta_percent = bits / flyhash_pool_size
            fly = FlyHash(mem_emb.shape[1], hash_dim=flyhash_pool_size, sparsity=0.10,
                           wta_percent=wta_percent, seed=seed)
            sim = SimHash(mem_emb.shape[1], hash_dim=bits, seed=seed)

            fly_codes = fly.hash(mem_emb)
            fly_queries = fly.hash(query_emb)
            sim_codes = sim.hash(mem_emb)
            sim_queries = sim.hash(query_emb)

            fly_dists = vectorized_hamming_ranking(fly_codes, fly_queries)
            sim_dists = vectorized_hamming_ranking(sim_codes, sim_queries)

            results[bits]["fly"].append(compute_map_labeled(fly_dists, relevant_sets))
            results[bits]["sim"].append(compute_map_labeled(sim_dists, relevant_sets))

        print(f"  seed {seed+1}/{n_seeds} done")

    print(f"\n{'k (bits)':>9} | {'FlyHash mAP (labeled)':>22} | {'SimHash mAP (labeled)':>22} | "
          f"{'gap > 2*std?':>12}")
    print("-" * 75)

    rows = []
    for bits in bit_budgets:
        fly_vals = np.array(results[bits]["fly"])
        sim_vals = np.array(results[bits]["sim"])
        fly_mean, fly_std = fly_vals.mean(), fly_vals.std()
        sim_mean, sim_std = sim_vals.mean(), sim_vals.std()
        gap = fly_mean - sim_mean
        combined_std = np.sqrt(fly_std**2 + sim_std**2)
        robust = "YES" if gap > 2 * combined_std else "no"

        print(f"{bits:>9} | {fly_mean:>14.3f}+/-{fly_std:<6.3f} | "
              f"{sim_mean:>14.3f}+/-{sim_std:<6.3f} | {robust:>12}")

        rows.append({"bits": bits, "fly_mean": fly_mean, "fly_std": fly_std,
                      "sim_mean": sim_mean, "sim_std": sim_std, "gap": gap, "robust": robust})

    print(f"\nCorpus size: {len(memories)} memories ('thousands', per the supervisor's ask).")
    print("Relevance labels are CONSTRUCTED (which slot/value each sentence encodes),")
    print("not derived from embedding nearest-neighbors — a genuine query-to-memory")
    print("relevance label, unlike every other dataset in this project.")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_personas", type=int, default=150)
    parser.add_argument("--n_query_slots_per_persona", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    run(n_personas=args.n_personas,
        n_query_slots_per_persona=args.n_query_slots_per_persona,
        embed_dim=args.embed_dim, n_seeds=args.seeds)
