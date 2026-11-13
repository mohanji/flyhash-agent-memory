"""Per-category retrieval recall on LoCoMo: FlyHash vs SimHash vs exact cosine, at matched bits.

For every conversation the script embeds each dialogue turn and each question, retrieves the top-k turns with
(a) FlyHash (score = shared active units), (b) SimHash (score = agreeing bits) and (c) exact cosine, and measures
how many of the question's annotated evidence turns come back. It needs no LLM and no API key.

Metrics (per question, then averaged within each question category):
  recall@k : fraction of the evidence turns that appear in the top-k
  all@k    : 1 if EVERY evidence turn appears in the top-k, else 0 (matters for multi-hop questions)

Fair-comparison choices:
  * FlyHash with k active bits is compared with SimHash with k bits (the paper's matched-bit protocol).
  * Ties are broken at random (seeded), never by turn order.
  * Several hash seeds are run; FlyHash/SimHash mean and std are over seeds. Exact cosine is deterministic.
  * `--center both` runs every method on raw and on mean-centred embeddings, so no method is favoured.

Usage (from the repo root, after downloading locomo10.json from the official LoCoMo repo):
    pip install numpy sentence-transformers
    python src/locomo_recall.py --data locomo10.json --embedder minilm --bits 16 32 64 128 --k 5 10 --seeds 10
    python src/locomo_recall.py --data locomo10.json --embedder bow      # quick smoke test, no model download

Category numbers follow the LoCoMo release as commonly documented (1 multi-hop, 2 temporal, 3 open-domain,
4 single-hop, 5 adversarial). The script prints one example question per category so you can check the mapping
yourself, and `--category-names` lets you override it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyhash import FlyHash, SimHash  # noqa: E402

DEFAULT_CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
DIA_RE = re.compile(r"D\d+:\d+")


# ----------------------------------------------------------------------------------------------- data
def load_locomo(path):
    data = json.load(open(path, encoding="utf-8"))
    convs, skipped = [], defaultdict(int)
    for si, sample in enumerate(data):
        sid = sample.get("sample_id", f"conv-{si}")
        conv = sample["conversation"]
        keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
        turns = []
        for sk in keys:
            for t in conv[sk]:
                text = t.get("text", "")
                if t.get("blip_caption"):
                    text = f"{text} [shared image: {t['blip_caption']}]"
                turns.append({"dia_id": t["dia_id"], "text": f"{t.get('speaker', '')}: {text}".strip()})
        index = {t["dia_id"]: i for i, t in enumerate(turns)}
        qas = []
        for qi, qa in enumerate(sample.get("qa", [])):
            raw = qa.get("evidence") or []
            raw = [raw] if isinstance(raw, str) else raw
            ids = []
            for s in raw:                                  # evidence strings can be "D8:6; D9:17" or have stray text
                ids.extend(DIA_RE.findall(str(s)))
            if not ids:
                skipped["no evidence listed"] += 1
                continue
            ev = sorted({index[d] for d in ids if d in index})
            if not ev:
                skipped["evidence id not found in conversation"] += 1
                continue
            qas.append({"qid": f"{sid}/{qi}", "question": qa["question"], "category": int(qa.get("category", 0)), "evidence": ev})
        convs.append({"sid": sid, "turns": turns, "qas": qas})
    return convs, skipped


class Embedder:
    def __init__(self, how, model):
        self.how = how
        if how == "minilm":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model)

    def __call__(self, texts):
        texts = list(texts)
        if self.how == "minilm":
            return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=64,
                                                show_progress_bar=False), dtype=np.float32)
        out = np.zeros((len(texts), 384), dtype=np.float32)      # hashed bag of words, smoke tests only
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9']+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 384] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def centre(T, Q):
    mu = T.mean(axis=0, keepdims=True)
    T, Q = T - mu, Q - mu
    nt = np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    nq = np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
    return T / nt, Q / nq


# ------------------------------------------------------------------------------------------- retrieval
def topk(S, kmax, rng, tie_noise):
    """Indices of the top-kmax columns per row, best first; ties broken at random."""
    kmax = min(kmax, S.shape[1])
    if tie_noise:
        S = S + rng.random(S.shape, dtype=np.float32) * tie_noise
    idx = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
    part = np.take_along_axis(S, idx, 1)
    return np.take_along_axis(idx, np.argsort(-part, axis=1), 1)


def score(method, h, T, Q):
    if method == "dense":
        return Q @ T.T, 0.0
    ct, cq = h.hash(T).astype(np.float32), h.hash(Q).astype(np.float32)
    if method == "flyhash":
        return cq @ ct.T, 0.5                      # shared active units (integers) -> noise < 1 only breaks ties
    return cq @ ct.T + (1 - cq) @ (1 - ct).T, 0.5   # simhash: agreeing bits


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="locomo10.json")
    ap.add_argument("--embedder", choices=["minilm", "bow"], default="minilm")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="any sentence-transformers model")
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 32, 64, 128], help="FlyHash active bits = SimHash bits")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10], help="top-k values to report")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--hash-dim", type=int, default=2000, help="FlyHash expansion size")
    ap.add_argument("--center", choices=["no", "yes", "both"], default="both")
    ap.add_argument("--out", default="locomo_recall.csv")
    ap.add_argument("--dump-contexts", help="folder: write top-k contexts (JSONL, seed 0) for sharing")
    ap.add_argument("--dump-k", type=int, default=10)
    ap.add_argument("--dump-ids-only", action="store_true",
                    help="dump only LoCoMo turn ids (dia_id), no dialogue text, so no CC BY-NC text is redistributed; "
                         "the recipient maps ids to text with their own copy of locomo10.json")
    ap.add_argument("--print-k", type=int, help="k shown in the printed table (default: largest --k)")
    ap.add_argument("--category-names", help='JSON, e.g. \'{"1":"multi-hop","2":"temporal"}\'')
    ap.add_argument("--exclude-from-all", type=int, nargs="*", default=[5],
                    help="categories left out of the 'all' column (default 5 = adversarial, which most LoCoMo "
                         "papers exclude because those questions are unanswerable); they still get their own column")
    ap.add_argument("--extra-simhash-bits", type=int, nargs="*", default=[],
                    help="additional SimHash-only bit counts (storage-matched control, e.g. 256 512 1024 2000). "
                         "FlyHash b active units in a 2000-unit space cost about b*11 bits as an index list or 2000 "
                         "bits as a bitmap, so SimHash at 1024-2000 bits is the equal-storage comparison")
    ap.add_argument("--bootstrap", type=int, default=2000, help="paired bootstrap resamples over questions (0 = off)")
    ap.add_argument("--boot-k", type=int, nargs="*", default=[10], help="k values for the bootstrap table")
    a = ap.parse_args()
    excl = set(a.exclude_from_all)

    names = dict(DEFAULT_CATEGORY_NAMES)
    if a.category_names:
        names.update({int(k): v for k, v in json.loads(a.category_names).items()})
    convs, skipped = load_locomo(a.data)
    nturn = sum(len(c["turns"]) for c in convs)
    nq = sum(len(c["qas"]) for c in convs)
    cats = defaultdict(int)
    examples = {}
    for c in convs:
        for q in c["qas"]:
            cats[q["category"]] += 1
            examples.setdefault(q["category"], q["question"])
    print(f"{len(convs)} conversations, {nturn} turns, {nq} questions with usable evidence; skipped: {dict(skipped)}")
    for cat in sorted(cats):
        print(f"  category {cat} ({names.get(cat, '?')}): {cats[cat]} questions, e.g. {examples[cat]!r}")
    print("Check that the category names match these examples.\n")

    emb = Embedder(a.embedder, a.model)
    seeds = list(range(a.seeds))
    kmax = max(max(a.k), a.dump_k if a.dump_contexts else 0)
    centers = {"no": [False], "yes": [True], "both": [False, True]}[a.center]
    hashers = {}
    acc = defaultdict(lambda: [0.0, 0.0, 0])        # (centre, method, bits, seed, cat, k) -> [sum recall, sum all, n]
    dumps = {}
    starts, t0 = [], 0
    for c in convs:
        starts.append(t0)
        t0 += len(c["qas"])
    qcat = np.array([q["category"] for c in convs for q in c["qas"]])
    qconv = np.array([ci for ci, c in enumerate(convs) for _ in c["qas"]])
    qrec = defaultdict(lambda: np.zeros(t0))        # (centre, method, bits, k) -> per-question recall, seed-averaged

    def add(key, rec, allhit):
        for cat in [key[4]] + ([] if key[4] in excl else ["all"]):
            v = acc[key[:4] + (cat, key[5])]
            v[0] += rec
            v[1] += allhit
            v[2] += 1

    for ci, c in enumerate(convs):
        turns, qas = c["turns"], c["qas"]
        if not qas:
            continue
        T0 = emb([t["text"] for t in turns])
        Q0 = emb([q["question"] for q in qas])
        for cen in centers:
            T, Q = centre(T0, Q0) if cen else (T0, Q0)
            dim = T.shape[1]
            sim_bits = list(a.bits) + [b for b in a.extra_simhash_bits if b not in a.bits]
            configs = ([("dense", 0, 0)] + [("flyhash", b, s) for b in a.bits for s in seeds]
                       + [("simhash", b, s) for b in sim_bits for s in seeds])
            for method, bits, seed in configs:
                h = None
                if method != "dense":
                    hk = (cen, method, bits, seed)
                    if hk not in hashers:
                        hashers[hk] = (FlyHash(dim, a.hash_dim, wta_percent=bits / a.hash_dim, seed=seed)
                                       if method == "flyhash" else SimHash(dim, bits, seed=seed))
                    h = hashers[hk]
                S, tie = score(method, h, T, Q)
                rng = np.random.default_rng(1000 * ci + 7 * seed + bits)
                R = topk(S, kmax, rng, tie)
                for qi, q in enumerate(qas):
                    ev = set(q["evidence"])
                    for k in a.k:
                        got = len(ev & set(R[qi, :k].tolist()))
                        add((cen, method, bits, seed, q["category"], k), got / len(ev), float(got == len(ev)))
                        qrec[(cen, method, bits, k)][starts[ci] + qi] += (got / len(ev)) / (1 if method == "dense" else len(seeds))
                if a.dump_contexts and seed == 0 and cen == centers[-1]:
                    Path(a.dump_contexts).mkdir(parents=True, exist_ok=True)
                    fn = Path(a.dump_contexts) / f"{method}{'' if method == 'dense' else '-b' + str(bits)}.jsonl"
                    if fn not in dumps:
                        dumps[fn] = open(fn, "w", encoding="utf-8")
                    for qi, q in enumerate(qas):
                        rec = {"qid": q["qid"], "category": q["category"],
                               "evidence_dia_ids": [turns[i]["dia_id"] for i in q["evidence"]],
                               "ranked_dia_ids": [turns[i]["dia_id"] for i in R[qi, :a.dump_k].tolist()]}  # best first: top-5 = first 5
                        if not a.dump_ids_only:
                            idx = sorted(R[qi, :a.dump_k].tolist())         # chronological order for a reader
                            rec["memories"] = [{"text": turns[i]["text"], "dia_id": turns[i]["dia_id"],
                                                "score": float(S[qi, i])} for i in idx]
                        dumps[fn].write(json.dumps(rec) + "\n")
        print(f"conversation {ci + 1}/{len(convs)} done ({c['sid']}, {len(turns)} turns, {len(qas)} questions)",
              file=sys.stderr)
    for f in dumps.values():
        f.close()

    # aggregate over seeds
    rows = []
    groups = defaultdict(dict)
    for (cen, method, bits, seed, cat, k), (sr, sa, n) in acc.items():
        groups[(cen, method, bits, cat, k)][seed] = (sr / n, sa / n, n)
    for (cen, method, bits, cat, k), per in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], str(x[0][3]), x[0][4])):
        rec = np.array([v[0] for v in per.values()])
        alh = np.array([v[1] for v in per.values()])
        rows.append({"centered": cen, "method": method, "bits": bits, "category": cat, "k": k,
                     "recall_mean": rec.mean(), "recall_std": rec.std(), "all_mean": alh.mean(), "all_std": alh.std(),
                     "n_questions": next(iter(per.values()))[2], "n_seeds": len(per)})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")

    pk = a.print_k or max(a.k)
    cols = ["all"] + [c for c in sorted(cats)]
    for cen in centers:
        for metric, label in (("recall", "recall@%d" % pk), ("all", "all-evidence@%d" % pk)):
            print(f"\n=== {label}, {'mean-centred' if cen else 'raw'} embeddings ({a.embedder}), mean±std over {a.seeds} seeds ===")
            allname = "all*" if excl else "all"
            print(f"{'method':<14}" + "".join(f"{allname if c == 'all' else names.get(c, c):>16}" for c in cols))
            seen = []
            for r in rows:
                if r["centered"] != cen or r["k"] != pk:
                    continue
                key = (r["method"], r["bits"])
                if key not in seen:
                    seen.append(key)
            for method, bits in seen:
                line = f"{method + ('' if method == 'dense' else ' b=' + str(bits)):<14}"
                for c in cols:
                    r = next((x for x in rows if x["centered"] == cen and x["method"] == method and x["bits"] == bits
                              and x["category"] == c and x["k"] == pk), None)
                    cell = f"{r[metric + '_mean']:.3f}±{r[metric + '_std']:.3f}" if r else "-"
                    line += f"{cell:>16}"
                print(line)
            if excl:
                print("* 'all*' excludes: " + ", ".join(f"{c} ({names.get(c, '?')})" for c in sorted(excl)))

    # ---------------------------------------------------------------- paired bootstrap over questions
    if a.bootstrap > 0:
        pool = np.where(~np.isin(qcat, list(excl)))[0]
        brng = np.random.default_rng(12345)
        bidx = brng.integers(0, len(pool), (a.bootstrap, len(pool)))
        brows = []

        def compare(label, cen, k, m1, b1, m2, b2):
            x, y = qrec.get((cen, m1, b1, k)), qrec.get((cen, m2, b2, k))
            if x is None or y is None:
                return
            d = (x - y)[pool]
            bm = d[bidx].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            cd = qconv[pool]
            wins = sum(d[cd == ci].mean() > 0 for ci in range(len(convs)) if (cd == ci).any())
            brows.append({"centered": cen, "k": k, "comparison": label, "a": f"{m1}{b1 or ''}", "b": f"{m2}{b2 or ''}",
                          "diff": d.mean(), "ci_lo": lo, "ci_hi": hi, "convs_a_better": wins, "n_convs": len(convs),
                          "n_questions": len(pool)})

        sim_all = sorted({b for (cn, m, b, kk) in qrec if m == "simhash"})
        fly_all = sorted({b for (cn, m, b, kk) in qrec if m == "flyhash"})
        for cen in centers:
            for k in [k for k in a.boot_k if k in a.k]:
                for b in fly_all:
                    compare("flyhash - simhash, same bits", cen, k, "flyhash", b, "simhash", b)
                    compare("flyhash - dense", cen, k, "flyhash", b, "dense", 0)
                for fb in fly_all:                      # equal-storage controls: SimHash with >= FlyHash's bit count
                    for sb in sim_all:
                        if sb > fb:
                            compare("flyhash - simhash, more simhash bits", cen, k, "flyhash", fb, "simhash", sb)
        bout = Path(a.out).with_name(Path(a.out).stem + "_bootstrap.csv")
        if brows:
            with open(bout, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(brows[0]))
                w.writeheader()
                w.writerows(brows)
            print(f"\nwrote {bout}")
            print(f"\n=== paired bootstrap over {len(pool)} questions ({a.bootstrap} resamples; "
                  f"conversations: how many of {len(convs)} favour the first method) ===")
            for r in brows:
                print(f"{'centred' if r['centered'] else 'raw':<8} k={r['k']:<3} {r['a']:>11} - {r['b']:<11} "
                      f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  convs {r['convs_a_better']}/{r['n_convs']}")
            print("* CI = 95% percentile interval over questions (hash-seed variance already averaged out); "
                  "questions in one conversation are correlated, so read the conv count as a robustness check.")


        # ---- per category: FlyHash recall as a share of dense recall (ratio of means), bootstrap within category.
        # Categories are disjoint sets of questions, so their resamples are independent; the difference of two
        # categories' shares is bootstrapped from those independent resamples.
        crows, cat_ids = [], [c for c in sorted(cats) if c not in excl]
        for cen in centers:
            for k in [k for k in a.boot_k if k in a.k]:
                dn = qrec.get((cen, "dense", 0, k))
                if dn is None:
                    continue
                for b in fly_all:
                    fl = qrec.get((cen, "flyhash", b, k))
                    if fl is None:
                        continue
                    shares = {}
                    for c in cat_ids:
                        idx = np.where(qcat == c)[0]
                        bi = brng.integers(0, len(idx), (a.bootstrap, len(idx)))
                        s = fl[idx][bi].mean(axis=1) / np.maximum(dn[idx][bi].mean(axis=1), 1e-12)
                        shares[c] = s
                        lo, hi = np.percentile(s, [2.5, 97.5])
                        crows.append({"centered": cen, "k": k, "bits": b, "category": names.get(c, str(c)),
                                      "flyhash": fl[idx].mean(), "dense": dn[idx].mean(),
                                      "share_of_dense": fl[idx].mean() / dn[idx].mean(), "ci_lo": lo, "ci_hi": hi,
                                      "n_questions": len(idx)})
                    ref = 4 if 4 in shares else None
                    for c in cat_ids:
                        if ref is None or c == ref:
                            continue
                        d = shares[c] - shares[ref]
                        lo, hi = np.percentile(d, [2.5, 97.5])
                        crows.append({"centered": cen, "k": k, "bits": b,
                                      "category": f"{names.get(c, c)} minus {names.get(ref, ref)} (share of dense)",
                                      "share_of_dense": d.mean(), "ci_lo": lo, "ci_hi": hi})
        if crows:
            cout = Path(a.out).with_name(Path(a.out).stem + "_percat_bootstrap.csv")
            cols = ["centered", "k", "bits", "category", "flyhash", "dense", "share_of_dense", "ci_lo", "ci_hi", "n_questions"]
            with open(cout, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(crows)
            print(f"\nwrote {cout}")
            print("\n=== FlyHash recall as a share of dense recall, by category (95% bootstrap interval within category) ===")
            for r in crows:
                if r["centered"]:
                    print(f"k={r['k']:<3} {r['bits']:>4} active bits  {r['category']:<42} "
                          f"{r['share_of_dense']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
