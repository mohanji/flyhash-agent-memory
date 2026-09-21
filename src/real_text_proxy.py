"""
real_text_proxy.py — Validates the "FlyHash needs correlated features"
hypothesis RIGHT NOW, without network access, using real (if small)
natural-language text embedded via TF-IDF instead of sentence-transformers.

Why this stands in for real_embeddings.py: TF-IDF vectors over real
sentences have genuine word-co-occurrence structure (many dimensions are
correlated — "protein" and "muscle" tend to activate together), unlike the
isotropic Gaussian blobs used in benchmark.py. That correlated structure is
exactly the condition under which FlyHash's sparse random projection is
theorized to preserve more neighbor information than SimHash's dense one.

Run real_embeddings.py on your own machine (with internet) for the full
version with actual sentence-transformer embeddings; this script exists so
you can check the logic and see a directional result today.
"""

from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from real_embeddings import SAMPLE_CORPUS
from recall_vs_bits import vectorized_hamming_ranking, recall_at_k_from_ranking
from flyhash import FlyHash, SimHash


def load_real_text_embeddings(n_docs: int = 500, embed_dim: int = 128, seed: int = 0):
    """Expands SAMPLE_CORPUS to n_docs and embeds with TF-IDF -> SVD."""
    rng = np.random.default_rng(seed)
    base = SAMPLE_CORPUS
    # repeat + shuffle word order slightly isn't needed; repetition alone
    # still gives TF-IDF meaningful (if repetitive) document vectors, and
    # keeps this dependency-free (no extra text augmentation library).
    texts = [base[i % len(base)] for i in range(n_docs)]

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(texts)
    n_components = min(embed_dim, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    embeddings = svd.fit_transform(X).astype(np.float32)
    return texts, embeddings


def run(n_docs=500, embed_dim=128, n_queries=100, k=5,
        bit_budgets=(64, 128, 256, 512)):
    texts, embeddings = load_real_text_embeddings(n_docs, embed_dim)
    print(f"Corpus: {len(texts)} texts (from {len(SAMPLE_CORPUS)} base sentences, "
          f"repeated), embedding dim after SVD: {embeddings.shape[1]}")

    rng = np.random.default_rng(1)
    query_idx = rng.choice(len(texts), size=min(n_queries, len(texts)), replace=False)
    queries = embeddings[query_idx]

    diffs = queries[:, None, :] - embeddings[None, :, :]
    exact_dists = np.linalg.norm(diffs, axis=2)

    print(f"\n{'bits':>6} | {'FlyHash recall@'+str(k):>16} | {'SimHash recall@'+str(k):>16}")
    print("-" * 46)

    for bits in bit_budgets:
        fly = FlyHash(embeddings.shape[1], hash_dim=bits, sparsity=0.10, wta_percent=0.05)
        fly_codes = fly.hash(embeddings)
        fly_queries = fly.hash(queries)

        sim = SimHash(embeddings.shape[1], hash_dim=bits)
        sim_codes = sim.hash(embeddings)
        sim_queries = sim.hash(queries)

        fly_dists = vectorized_hamming_ranking(fly_codes, fly_queries)
        sim_dists = vectorized_hamming_ranking(sim_codes, sim_queries)

        fly_recall = recall_at_k_from_ranking(fly_dists, exact_dists, k)
        sim_recall = recall_at_k_from_ranking(sim_dists, exact_dists, k)

        print(f"{bits:>6} | {fly_recall:>16.3f} | {sim_recall:>16.3f}")


if __name__ == "__main__":
    run()
