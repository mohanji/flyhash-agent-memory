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
  flyhash.py          FlyHash + SimHash implementations
  memory_store.py     Agent-memory-store wrapper (FlyHash-backed + FAISS baseline)
  benchmark.py         End-to-end memory-store benchmark (recall / latency / memory)
  recall_vs_bits.py    Fair "equal bit budget" FlyHash vs SimHash comparison
  real_embeddings.py   Real sentence-transformers loader (100 unique sentences, 5 topics)
  real_text_proxy.py   Offline stand-in using TF-IDF over real text (no network needed)
  agent_memory.py      Text-in/text-out memory wrapper: embeds + calls FlyMemoryStore
  agent_demo.py         Terminal chat agent (Claude API) using AgentMemory as a tool
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

**Equal bit-budget comparison, real sentence-transformer embeddings**
(`recall_vs_bits.py` with `real_embeddings.py` as the loader, 100 unique
sentences across 5 topics, `all-MiniLM-L6-v2`, 384-dim) — SimHash wins
again, decisively:

| Bits | FlyHash recall@10 | SimHash recall@10 |
|---|---|---|
| 128  | 0.340 | 0.548 |
| 512  | 0.488 | 0.714 |
| 2048 | 0.670 | 0.852 |

**The three-way finding, and why it matters more than any single number:**

| Data type | Winner | Likely reason |
|---|---|---|
| Isotropic synthetic blobs | SimHash | No redundancy for FlyHash's sparse projection to exploit |
| TF-IDF over real text (non-negative, redundant) | **FlyHash** | Matches the input assumptions the biological circuit was built around |
| Dense signed sentence-transformer embeddings | SimHash | Violates FlyHash's implicit non-negativity assumption; deep embeddings are already fairly decorrelated, so k-WTA discards useful signal rather than redundancy |

The fly's real olfactory circuit operates on non-negative firing rates
(a neuron can't fire a negative amount). Sentence-transformer embeddings
are dense, signed, real-valued vectors — nothing like a firing rate — so
FlyHash's "keep only the most active units" logic, which is exactly what
lets it discard redundancy in TF-IDF-style non-negative data, instead
discards genuinely useful signal in this dense signed regime. This project
does not conclude "FlyHash is worse"; it identifies **the specific
condition FlyHash's advantage depends on**, which is a stronger and more
useful conclusion than a single clean win would have been.

A natural follow-up (not yet implemented, listed in Roadmap): split each
embedding into positive/negative parts before hashing (`x⁺ = max(x,0)`,
`x⁻ = max(-x,0)`, concatenated), which is the standard trick for adapting
non-negativity-assuming algorithms to signed data, and may recover
FlyHash's advantage even on dense sentence embeddings.

## Honest findings (this matters more than a clean win for a resume project)

1. **FlyHash's advantage over SimHash depends on a specific input
   condition — non-negative, redundant features — not on "being
   bio-inspired."** It wins on TF-IDF-style non-negative text data, loses
   on isotropic synthetic data, and loses again on dense signed
   sentence-transformer embeddings. Reproducing and explaining *when* a
   biologically-inspired method's claimed advantage does and doesn't hold
   is the actual scientific content of a project like this — a blanket
   win would have been less informative, not more.
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

## Agent integration (Day 4)

`agent_memory.py` wraps `FlyMemoryStore` behind a plain text-in/text-out
interface (`remember(text)` / `recall(query, k)`), so an agent never has
to touch embeddings or hashing directly. `agent_demo.py` is a small
terminal chat agent using the Claude API with two tools:

- `save_memory` — the model calls this when the user shares a fact worth
  remembering (a preference, constraint, personal detail).
- `recall_memory` — the model calls this when answering something that
  might depend on a fact from earlier in the conversation.

Setup:
```bash
pip install anthropic
# Windows (new terminal after):
setx ANTHROPIC_API_KEY "your-key-here"
# macOS/Linux:
export ANTHROPIC_API_KEY="your-key-here"

cd src
python agent_demo.py
```

Try this to see cross-turn recall in action:
```
You: My favorite language is Python and I'm allergic to peanuts.
You: What's a good weekend project idea?
You: What should I avoid eating?
```
The agent should correctly answer "peanuts" on the third turn, even
though that fact was never mentioned in that turn — it has to call
`recall_memory` and retrieve it from a prior turn via the FlyHash store.

## Roadmap (next steps)

- [x] Swap synthetic blobs for real, correlated text — validated with an
      offline TF-IDF proxy (`real_text_proxy.py`) AND real
      `sentence-transformers` embeddings (`real_embeddings.py`); found
      that FlyHash's advantage over SimHash depends on non-negative,
      redundant input structure — it wins on TF-IDF, loses on isotropic
      synthetic data, and loses again on dense signed sentence embeddings.
- [ ] Try splitting embeddings into positive/negative parts before hashing
      (`x⁺ = max(x,0)`, `x⁻ = max(-x,0)`, concatenated) to see whether this
      recovers FlyHash's advantage on dense signed embeddings — the
      standard trick for adapting non-negativity-assuming algorithms to
      signed data.
- [ ] Grid-search `sparsity` and `wta_percent` — the fly's own ratios
      (~12% input sparsity, ~5% winner-take-all) may not be optimal for
      embeddings very different in dimensionality from ~50 ORNs.
- [ ] Implement real multi-probe/banded LSH indexing on top of the sparse
      codes to get actual sublinear query time at scale (this is the
      piece needed before claiming a latency win over FAISS).
- [x] Wire `FlyMemoryStore` into an actual agent loop (`agent_memory.py` +
      `agent_demo.py`, Claude API tool-use) — demonstrated cross-turn
      recall of facts shared earlier in a conversation.
- [ ] Package results into a short write-up: motivation, method, honest
      results table, and what you'd do with more time — this is the
      artifact to link from your resume/portfolio.


