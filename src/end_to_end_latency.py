"""
end_to_end_latency.py — supplements the Section 4.5 agent-memory demo with
one measured number: full pipeline latency (embedding + FlyHash hash +
rank lookup), not just the isolated hash-comparison timings of
Sections 4.3/4.4. This does NOT replace those sections' matched-active-bit
comparisons (embedding time is identical regardless of which hash is used
downstream, so it has no bearing on the FlyHash-vs-SimHash comparison) —
it exists only to put a number on the limitation noted in Section 6
("we do not report memory/latency figures for the full end-to-end agent
memory system"), so a reader can see the full picture was checked, not
just assumed to be fine.

Hash-comparison stage: uses the SPARSE intersection-count method from
sparse_flyhash_benchmark.py (Section 4.4's "sparse-set comparison"),
i.e. a scipy sparse matrix dot product, rather than FlyMemoryStore's
"scan" mode, which loops over records in Python with set intersections
and does NOT represent FlyHash's achievable speed (it was written for
correctness, not performance). Using the same sparse method as the rest
of the paper's speed numbers keeps this end-to-end figure consistent
with Table 5 rather than materially slower than it for implementation
reasons that have nothing to do with the algorithm.

Uses the same real sentence-transformer (all-MiniLM-L6-v2) as the
Section 3.5/4.5 agent-memory demo, and the realistic 6,000-memory corpus
from agent_memory_dataset.py (Section 4.7) as a representative store size.

Usage:
    python end_to_end_latency.py
"""

from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer

from agent_memory_dataset import build_dataset
from flyhash import FlyHash


def run(n_personas: int = 150, n_repeats: int = 30, k_retrieve: int = 3,
        hash_dim: int = 2000, sparsity: float = 0.10, wta_percent: float = 0.05):
    memories, queries, _relevant_sets = build_dataset(n_personas=n_personas)
    print(f"Corpus: {len(memories)} memories | {len(queries)} example queries\n")

    print("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_dim = model.get_sentence_embedding_dimension()

    print("Embedding and hashing corpus (sparse representation)...")
    mem_embeddings = model.encode(memories, show_progress_bar=False)
    hasher = FlyHash(embed_dim, hash_dim=hash_dim, sparsity=sparsity,
                      wta_percent=wta_percent, seed=42)
    mem_codes = hasher.hash(np.asarray(mem_embeddings, dtype=np.float32))
    mem_codes_sparse = sparse.csr_matrix(mem_codes)
    k_active = hasher.k

    query_text = queries[0]
    print(f"\nTiming query: {query_text!r}\n")

    # Warm up (first call pays one-time model/cache overhead).
    _ = model.encode([query_text])

    embed_times, hash_rank_times, total_times = [], [], []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        q_emb = model.encode([query_text])[0]
        t1 = time.perf_counter()

        q_code = hasher.hash(np.asarray(q_emb, dtype=np.float32)[None, :])
        q_sparse = sparse.csr_matrix(q_code)
        intersection = mem_codes_sparse.dot(q_sparse.T).toarray().ravel()
        dists = 2 * k_active - 2 * intersection  # Hamming distance, fixed-k identity
        top_k_idx = np.argsort(dists)[:k_retrieve]
        _ = [(memories[i], int(dists[i])) for i in top_k_idx]

        t2 = time.perf_counter()

        embed_times.append((t1 - t0) * 1000)
        hash_rank_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)

    embed_ms = float(np.mean(embed_times))
    hash_rank_ms = float(np.mean(hash_rank_times))
    total_ms = float(np.mean(total_times))

    print(f"{'Stage':>28} | {'ms/query':>10} | {'% of total':>10}")
    print("-" * 55)
    print(f"{'Sentence embedding (MiniLM)':>28} | {embed_ms:>10.3f} | "
          f"{100 * embed_ms / total_ms:>9.1f}%")
    print(f"{'FlyHash hash + sparse rank':>28} | {hash_rank_ms:>10.3f} | "
          f"{100 * hash_rank_ms / total_ms:>9.1f}%")
    print(f"{'Total end-to-end':>28} | {total_ms:>10.3f} | {'100.0%':>10}")

    print(f"\nCorpus size: {len(memories)} records, hash_dim={hash_dim}, "
          f"k_active={k_active} (wta_percent={wta_percent}).")
    print("Hash-comparison stage uses the SPARSE intersection method (Section 4.4),")
    print("the same method behind Table 5's speed numbers, not the unoptimized")
    print("Python-loop 'scan' mode used in an earlier version of this script.")
    print("Embedding time is identical regardless of which hash (FlyHash or")
    print("SimHash) is used downstream, so it does not bear on the")
    print("FlyHash-vs-SimHash comparisons in Sections 4.1-4.4/4.6-4.7; it is")
    print("reported here only to give the full pipeline's latency a real")
    print("number, per Section 6's limitation.")

    return {"embed_ms": embed_ms, "hash_rank_ms": hash_rank_ms, "total_ms": total_ms,
            "n_records": len(memories), "k_active": k_active}


if __name__ == "__main__":
    run()
