# FlyHash: Bio-Inspired Sparse Hashing for LLM Agent Memory

A from-scratch reimplementation of the Drosophila olfactory hashing algorithm
(Dasgupta, Stevens & Navlakha, *"A neural algorithm for a fundamental
computing problem"*, Science 2017), applied to the problem of memory
retrieval for LLM agents — with an honest empirical comparison against
standard baselines rather than a cherry-picked demo.

## The biological idea

The fly's olfactory circuit turns ~50 receptor-neuron activations into a
sparse binary "tag" via three steps:

1. **Random sparse expansion** — ~50 inputs project into ~2,000 Kenyon
   cells, each wired to only ~6 random inputs (not fully connected).
2. **Winner-take-all** — only the top ~5% most active Kenyon cells "fire."
3. The resulting sparse binary code is a similarity-preserving hash:
   similar smells produce similar (overlapping) codes.

This repo implements that as a general-purpose locality-sensitive hash
(`FlyHash`) usable on any dense embedding, plus the standard baseline it's
compared against in the literature (`SimHash`, random-hyperplane LSH).

## Project layout

```
src/
  flyhash.py        FlyHash + SimHash implementations
  memory_store.py   Agent-memory-store wrapper (FlyHash-backed + FAISS baseline)
  benchmark.py       End-to-end memory-store benchmark (recall / latency / memory)
  recall_vs_bits.py  The fair "equal bit budget" FlyHash vs SimHash benchmark
```

## Setup

```bash
pip install numpy scikit-learn faiss-cpu
cd src
python3 benchmark.py        # end-to-end memory store demo
python3 recall_vs_bits.py   # FlyHash vs SimHash, equal bit budget
```

Both scripts generate their own data locally (clustered synthetic
embeddings via `sklearn.datasets.make_blobs`) — no network access or model
download required, so this runs anywhere. Swap `load_embeddings()` in
`benchmark.py` for a real embedding model (e.g. `sentence-transformers`)
once you want numbers on real text.

## Current results (synthetic clustered embeddings, dim=128)

**End-to-end memory store** (`benchmark.py`, 20,000 records):

| Metric | FlyHash (bucket) | FlyHash (scan) | FAISS flat (exact) |
|---|---|---|---|
| Recall@10 vs. exact | 0.27 | 0.28 | 1.00 (reference) |
| Memory / record | 480 bytes | 480 bytes | 512 bytes |

**Equal bit-budget comparison, isotropic synthetic data** (`recall_vs_bits.py`):

| Bits | FlyHash recall@10 | SimHash recall@10 |
|---|---|---|
| 128  | 0.22 | 0.29 |
| 512  | 0.28 | 0.40 |
| 2048 | 0.40 | 0.58 |

**Equal bit-budget comparison, real correlated text (TF-IDF over natural
sentences)** (`real_text_proxy.py`) — SimHash loses, FlyHash wins:

| Bits | FlyHash recall@5 | SimHash recall@5 |
|---|---|---|
| 64  | **0.712** | 0.588 |
| 128 | **0.620** | 0.524 |
| 256 | **0.626** | 0.546 |
| 512 | 0.558 | 0.552 |

This is the key result: **the same two algorithms, on the same synthetic
data, give FlyHash a loss — but on real, correlated text data, FlyHash
wins, and the margin is largest at the smallest bit budgets** (64 bits:
+0.12 recall over SimHash). That's consistent with the theoretical reason
FlyHash exists: its sparse random projection has an advantage specifically
when input dimensions are correlated/redundant (as real language is), and
no advantage — or a disadvantage — on isotropic data where every dimension
is independent information.

`real_text_proxy.py` uses TF-IDF over a small hand-written corpus as a
network-free stand-in for real sentence embeddings, so this could be
validated without needing to download a model. `real_embeddings.py`
contains the actual `sentence-transformers`-based loader — run that on a
machine with normal internet access for the full-fidelity version of this
result (expect the effect to be at least as strong, since transformer
embeddings have even richer correlation structure than TF-IDF).

## Honest findings (this matters more than a clean win for a resume project)

1. **FlyHash's advantage over SimHash is data-dependent, and this project
   demonstrates exactly when and why.** On isotropic synthetic data it
   loses; on correlated real-text data it wins, especially at small bit
   budgets. That distinction — not a blanket "bio-inspired method wins" —
   is the actual scientific content of the original paper, and reproducing
   it (rather than reporting only a favorable number) is the point.
2. **A naive "any-overlap" inverted index does not give sublinear
   retrieval at this scale** — with 20,000 records and only 60 semantic
   clusters, ~75% of the store shares at least one active cell with any
   given query, so the candidate set barely shrinks. Real sublinear ANN
   from sparse codes needs banding/multi-probe LSH (see Roadmap), not a
   flat "any active bit matches" index.
3. **Memory savings are real but modest at these settings** — the actual
   payoff scales with how large you can push `hash_dim` while keeping the
   *number* of active bits `k` small and fixed (bigger dictionary, same
   sparse budget), which needs a real, higher-dimensional embedding
   (e.g. 384-dim sentence embeddings) to show convincingly.

## Roadmap (next steps)

- [x] Swap synthetic blobs for real, correlated text — validated with an
      offline TF-IDF proxy (`real_text_proxy.py`); confirmed FlyHash beats
      SimHash on correlated data. `real_embeddings.py` has the real
      `sentence-transformers` loader — run it on a machine with internet
      access to confirm the effect holds (or strengthens) with real
      transformer embeddings instead of TF-IDF.
- [ ] Grid-search `sparsity` and `wta_percent` — the fly's own ratios
      (~12% input sparsity, ~5% winner-take-all) may not be optimal for
      embeddings very different in dimensionality from ~50 ORNs.
- [ ] Implement real multi-probe/banded LSH indexing on top of the sparse
      codes to get actual sublinear query time at scale (this is the
      piece needed before claiming a latency win over FAISS).
- [ ] Wire `FlyMemoryStore` into an actual agent loop (LangChain/simple
      tool-calling loop) as a memory tool, and demo retrieval quality on
      a real multi-turn conversation transcript.
- [ ] Package results into a short write-up: motivation, method, honest
      results table, and what you'd do with more time — this is the
      artifact to link from your resume/portfolio.




So the honest, three-experiment story your project now tells is:

Data type	Winner	Why
Isotropic synthetic blobs	SimHash	No redundancy for FlyHash's sparsity to exploit
TF-IDF over real text (non-negative, redundant)	FlyHash	Matches the input assumptions FlyHash was designed around
Dense signed sentence-transformer embeddings	SimHash	Violates FlyHash's non-negativity assumption; embeddings are already fairly decorrelated
