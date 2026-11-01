"""
make_demo_data.py -- exports real FlyHash vs SimHash rankings for the interactive demo (docs/demo.html).

Queries are a fixed random sample (not hand-picked) from the released labeled benchmark. Everything shown in the demo
(active units, overlap counts, rankings, relevance flags, per-query AP) is computed here with the repo's own code,
seed 0, TF-IDF+SVD 256-d embeddings, pool size m=2000, matched active bits k.

Usage (repo root):  python scripts/make_demo_data.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from agent_memory_dataset import build_dataset
from agent_memory_realistic_benchmark import embed
from flyhash import FlyHash, SimHash
from recall_vs_bits import vectorized_hamming_ranking

KS = (16, 32, 64, 128)
M = 2000
TOP = 8
N_QUERIES = 12

memories, queries, rel = build_dataset(n_personas=150, seed=0)
mem_emb, q_emb = embed(memories, queries, embed_dim=256, seed=0)
rng = np.random.default_rng(11)
chosen = sorted(rng.choice(len(queries), size=N_QUERIES, replace=False).tolist())


def avg_prec(order, relevant):
    hits, s = 0, 0.0
    for r, i in enumerate(order, 1):
        if i in relevant:
            hits += 1; s += hits / r
            if hits == len(relevant): break
    return s / len(relevant)


out = {"m": M, "ks": list(KS), "top": TOP, "n_memories": len(memories),
       "grid": [50, 40], "queries": []}
for qi in chosen:
    out["queries"].append({"id": qi, "text": queries[qi],
                           "relevant": sorted(rel[qi]), "by_k": {}})
    relevant = rel[qi]
    for k in KS:
        fly = FlyHash(256, hash_dim=M, sparsity=0.10, wta_percent=k / M, seed=0)
        sim = SimHash(256, hash_dim=k, seed=0)
        fc, fq = fly.hash(mem_emb), fly.hash(q_emb[qi:qi + 1])
        sc, sq = sim.hash(mem_emb), sim.hash(q_emb[qi:qi + 1])
        fd = vectorized_hamming_ranking(fc, fq)[0]
        sd = vectorized_hamming_ranking(sc, sq)[0]
        # stable ordering so ties are deterministic
        f_order = np.argsort(fd, kind="stable"); s_order = np.argsort(sd, kind="stable")
        entry = {}
        for name, order, dist, qcode, codes in (("fly", f_order, fd, fq[0], fc), ("sim", s_order, sd, sq[0], sc)):
            top = []
            for i in order[:TOP].tolist():
                item = {"id": i, "text": memories[i], "rel": i in relevant,
                        "match": int(k - dist[i] / 2) if name == "fly" else int(k - dist[i])}
                if name == "fly":
                    item["active"] = np.flatnonzero(codes[i]).tolist()
                else:
                    item["bits"] = "".join(map(str, codes[i].tolist()))
                top.append(item)
            ranks = [r for r, i in enumerate(order.tolist(), 1) if i in relevant]
            entry[name] = {"top": top, "ap": round(avg_prec(order.tolist(), relevant), 4),
                           "relevant_ranks": ranks,
                           "query_active": np.flatnonzero(qcode).tolist() if name == "fly" else None,
                           "query_bits": "".join(map(str, qcode.tolist())) if name == "sim" else None}
        out["queries"][-1]["by_k"][str(k)] = entry

# full-benchmark table (paper, 10 seeds) for context
out["paper_map"] = {"k": [16, 32, 64, 128, 256], "fly": [.195, .271, .324, .365, .392],
                    "sim": [.038, .105, .212, .298, .345]}
path = os.path.join(ROOT, "docs", "demo_data.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
print("wrote", path, os.path.getsize(path) // 1024, "KB")
# honest summary of the sample
for k in KS:
    fa = np.mean([q["by_k"][str(k)]["fly"]["ap"] for q in out["queries"]])
    sa = np.mean([q["by_k"][str(k)]["sim"]["ap"] for q in out["queries"]])
    w = sum(q["by_k"][str(k)]["fly"]["ap"] > q["by_k"][str(k)]["sim"]["ap"] for q in out["queries"])
    l = sum(q["by_k"][str(k)]["fly"]["ap"] < q["by_k"][str(k)]["sim"]["ap"] for q in out["queries"])
    print(f"k={k}: sample mean AP fly={fa:.3f} sim={sa:.3f}  fly wins {w}, loses {l}, ties {N_QUERIES-w-l}")
