"""
labeled_equal_storage.py - equal-storage control for the labeled agent-memory benchmark.

Re-runs the paper's labeled benchmark (agent_memory_realistic_benchmark.py: same dataset, same TF-IDF+SVD
embedding, same Hamming ranking, same mAP) and adds what that script does not do:

  * SimHash at MORE bits than FlyHash has active bits (up to 2000), i.e. an equal-STORAGE comparison;
  * a dense cosine reference;
  * a paired bootstrap over queries, on seed-averaged per-query average precision.

Storage convention (same as the LoCoMo analysis): a FlyHash code with k active units out of m=2000 costs
about k*ceil(log2 m) = 11k bits as an index list, or m = 2000 bits as a bitmap. A SimHash code costs its bit count.

Put this file in src/ next to flyhash.py and agent_memory_realistic_benchmark.py.

Usage:
    python src/labeled_equal_storage.py                       # defaults: 10 seeds, FlyHash k = 16..256
    python src/labeled_equal_storage.py --seeds 10 --bootstrap 2000
"""

from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.decomposition import TruncatedSVD                      # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer         # noqa: E402

from agent_memory_dataset import build_dataset          # noqa: E402
from flyhash import FlyHash, SimHash                    # noqa: E402


def embed(memories, queries, embed_dim: int = 256, seed: int = 0):
    """Identical to agent_memory_realistic_benchmark.embed (copied to keep this script's imports small)."""
    tfidf = TfidfVectorizer()
    mem_X = tfidf.fit_transform(memories)
    query_X = tfidf.transform(queries)
    n_components = min(embed_dim, mem_X.shape[1] - 1, mem_X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return svd.fit_transform(mem_X).astype(np.float32), svd.transform(query_X).astype(np.float32)


def hamming(codes: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Hamming distances (q, n). float32 matmul (exact for these sizes), cast to int16 so that tie order under
    np.argsort matches the original recall_vs_bits.vectorized_hamming_ranking."""
    c, q = codes.astype(np.float32), queries.astype(np.float32)
    d = q.sum(1)[:, None] + c.sum(1)[None, :] - 2.0 * (q @ c.T)
    return np.rint(d).astype(np.int16)


def per_query_ap(dists: np.ndarray, relevant_sets: list[set]) -> np.ndarray:
    """Average precision per query over the full ranking (same definition as compute_map_labeled)."""
    ranking = np.argsort(dists, axis=1)
    aps = np.zeros(dists.shape[0])
    for qi, rel in enumerate(relevant_sets):
        if not rel:
            continue
        hits, psum = 0, 0.0
        for rank, idx in enumerate(ranking[qi], start=1):
            if idx in rel:
                hits += 1
                psum += hits / rank
                if hits == len(rel):
                    break
        aps[qi] = psum / len(rel)
    return aps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_personas", type=int, default=150)
    ap.add_argument("--n_query_slots_per_persona", type=int, default=1)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--fly-bits", type=int, nargs="+", default=[16, 32, 64, 128, 256], help="FlyHash active bits")
    ap.add_argument("--sim-bits", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2000])
    ap.add_argument("--pool", type=int, default=2000, help="FlyHash expansion size m")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default="labeled_equal_storage.csv")
    a = ap.parse_args()

    memories, queries, relevant_sets = build_dataset(n_personas=a.n_personas,
                                                     n_query_slots_per_persona=a.n_query_slots_per_persona, seed=0)
    nq = len(queries)
    print(f"{len(memories)} memories, {nq} queries, "
          f"{np.mean([len(s) for s in relevant_sets]):.2f} relevant memories per query")
    mem, qry = embed(memories, queries, embed_dim=a.embed_dim, seed=0)
    print(f"embedding dim {mem.shape[1]} (TF-IDF + SVD)\n")

    idx_bits = math.ceil(math.log2(a.pool))
    # (method, bits) -> list over seeds of per-query AP arrays
    aps: dict[tuple, list] = defaultdict(list)

    mn = mem / np.maximum(np.linalg.norm(mem, axis=1, keepdims=True), 1e-9)
    qn = qry / np.maximum(np.linalg.norm(qry, axis=1, keepdims=True), 1e-9)
    dense = per_query_ap(-(qn @ mn.T), relevant_sets)

    for seed in range(a.seeds):
        for k in a.fly_bits:
            fly = FlyHash(mem.shape[1], hash_dim=a.pool, sparsity=0.10, wta_percent=k / a.pool, seed=seed)
            aps[("flyhash", k)].append(per_query_ap(hamming(fly.hash(mem), fly.hash(qry)), relevant_sets))
        for b in a.sim_bits:
            sim = SimHash(mem.shape[1], hash_dim=b, seed=seed)
            aps[("simhash", b)].append(per_query_ap(hamming(sim.hash(mem), sim.hash(qry)), relevant_sets))
        print(f"  seed {seed + 1}/{a.seeds} done", file=sys.stderr)

    # ---- table over seeds
    rows = []
    print(f"{'method':<10}{'bits':>6}{'storage (bits)':>18}{'mAP mean':>10}{'std over seeds':>16}{'% of dense':>12}")
    print("-" * 72)
    dmean = dense.mean()
    print(f"{'dense':<10}{'-':>6}{'-':>18}{dmean:>10.3f}{'-':>16}{'100.0':>12}")
    for (method, b), lst in sorted(aps.items(), key=lambda x: (x[0][0], x[0][1])):
        per_seed = np.array([x.mean() for x in lst])
        storage = f"{idx_bits * b} idx / {a.pool} map" if method == "flyhash" else str(b)
        print(f"{method:<10}{b:>6}{storage:>18}{per_seed.mean():>10.3f}{per_seed.std():>16.3f}"
              f"{100 * per_seed.mean() / dmean:>12.1f}")
        rows.append({"method": method, "bits": b,
                     "storage_bits_index": idx_bits * b if method == "flyhash" else b,
                     "storage_bits_bitmap": a.pool if method == "flyhash" else b,
                     "map_mean": per_seed.mean(), "map_std": per_seed.std(), "dense_map": dmean})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")

    # ---- paired bootstrap over queries (seed-averaged per-query AP)
    if a.bootstrap > 0:
        avg = {key: np.mean(lst, axis=0) for key, lst in aps.items()}
        rng = np.random.default_rng(12345)
        bidx = rng.integers(0, nq, (a.bootstrap, nq))
        brows = []

        def cmp(fk, sb):
            d = avg[("flyhash", fk)] - avg[("simhash", sb)]
            bm = d[bidx].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            brows.append({"flyhash_active_bits": fk, "simhash_bits": sb, "diff_map": d.mean(),
                          "ci_lo": lo, "ci_hi": hi})

        for fk in a.fly_bits:
            for sb in a.sim_bits:
                if ("simhash", sb) in avg and sb >= fk:
                    cmp(fk, sb)
        print(f"\n=== paired bootstrap over {nq} queries ({a.bootstrap} resamples): "
              f"FlyHash(k active) - SimHash(b bits), mAP ===")
        for r in brows:
            tag = "same bits" if r["flyhash_active_bits"] == r["simhash_bits"] else "SimHash has more bits"
            print(f"FlyHash {r['flyhash_active_bits']:>4} - SimHash {r['simhash_bits']:>4}  "
                  f"{r['diff_map']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]   {tag}")
        bout = Path(a.out).with_name(Path(a.out).stem + "_bootstrap.csv")
        with open(bout, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(brows[0]))
            w.writeheader()
            w.writerows(brows)
        print(f"wrote {bout}")
        print("* queries come from templated personas, so intervals understate real-world variability; "
              "the benchmark is synthetic (see the paper's limitations).")


if __name__ == "__main__":
    main()
