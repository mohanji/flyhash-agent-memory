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
  recall_vs_bits.py    Corrected equal-hash-length FlyHash vs SimHash comparison
  map_metric.py        Mean Average Precision @ R, the literature-standard metric
  glove_embeddings.py  Real GloVe vector loader (used to find & fix the Phase 1 bug)
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

## Phase 1: the GloVe discrepancy, and how it was resolved

An earlier version of this README reported that FlyHash lost to SimHash on
every dataset tested, including real GloVe vectors — the same dataset the
original paper reports FlyHash *winning* on. That contradiction was
investigated properly rather than left unresolved or explained away:

1. **Ruled out normalization** — L2-normalizing inputs before hashing made
   no meaningful difference.
2. **Ruled out signed vs. true binary {0,1} projection weights** — matching
   the paper's exact projection matrix definition made no meaningful
   difference either.
3. **Ruled out evaluation metric** — switching from recall@k overlap to
   mean Average Precision (mAP@R), the metric the literature actually uses
   (confirmed via the BioHash paper, Ryali et al. 2020, which states
   explicitly that it follows Dasgupta et al.'s mAP protocol), gave the
   same qualitative picture.
4. **Found it** — diffing against the official reference implementation
   (github.com/dataplayer12/Fly-LSH) revealed the real bug: it was in this
   project's *experimental design*, not the algorithm. The paper's fair
   comparison holds the number of **winning bits** (`hash_length`, i.e. `k`
   in this repo) equal between FlyHash and conventional LSH — NOT the total
   hash dimensionality. FlyHash's Kenyon-cell pool size (`embedding_size`,
   `m`) is a separate, much larger parameter that winners are chosen from.
   The earlier benchmarks swept `hash_dim` directly while holding
   `wta_percent` fixed at 5%, which silently made FlyHash's *effective*
   code length (its number of active bits) a small fraction of what
   SimHash was being given at the same nominal "bits" value — an
   apples-to-oranges comparison, not a fair one.

**Corrected result, real GloVe vectors (glove-wiki-gigaword-100, 5000
words), mAP@100:**

| k (bits) | FlyHash mAP@100 | SimHash mAP@100 |
|---|---|---|
| 16  | **0.215** | 0.043 |
| 32  | **0.329** | 0.089 |
| 64  | **0.458** | 0.195 |
| 128 | **0.586** | 0.327 |
| 256 | **0.707** | 0.480 |

FlyHash now wins decisively at every bit budget, matching the original
paper's reported pattern. The same corrected protocol was also verified on
the synthetic isotropic dataset (`recall_vs_bits.py`, now fixed to hold `k`
constant against a fixed, large `flyhash_pool_size=2000`), where FlyHash
now wins there too — meaning the earlier "FlyHash only wins on redundant
data" narrative was itself an artifact of the flawed comparison, not a
real property of the data. **The non-negativity hypothesis from the
project's earlier stage was incorrect** — it was proposed to explain a gap
that turned out to be a bug in how the two methods were being compared,
not a genuine property of either algorithm.

## Honest findings (this matters more than a clean win for a resume project)

1. **A real, reproducible discrepancy with a published result was found,
   investigated systematically, and resolved to its actual root cause** —
   not through repeated hyperparameter guessing, but by ruling out
   candidate explanations one at a time and finally diffing against a
   reference implementation. That process, including the wrong hypothesis
   proposed and later retracted along the way, is documented above rather
   than cleaned up after the fact.
2. **Getting a "fair comparison" right is harder than it looks, and
   getting it wrong produces a confident-looking, entirely wrong
   conclusion** — the earlier version of this project had a coherent,
   plausible-sounding story (FlyHash needs non-negative redundant data)
   built entirely on a methodological bug. That's a useful lesson for
   evaluating any hashing/retrieval benchmark, not just this one.
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

- [x] Swap synthetic blobs for real, correlated text (`real_text_proxy.py`,
      `real_embeddings.py`) and real GloVe vectors (`glove_embeddings.py`).
- [x] **Phase 1 — find and fix the GloVe discrepancy.** Resolved: the bug
      was in the experimental design (comparing FlyHash's total hash
      dimension against SimHash's bit count, instead of holding the number
      of *winning* bits equal, per the paper's actual protocol). Confirmed
      fixed on both synthetic and real GloVe data; `real_embeddings.py`
      still needs the corrected protocol run on it on a machine with
      internet access (see Phase 1 section above for the exact numbers).
- [ ] **Phase 2 — real benchmark datasets + statistical rigor**: add SIFT1M
      or MNIST as a second established benchmark beyond GloVe; run 5-10
      random seeds per configuration and report mean ± std; grid-search
      `sparsity` and the FlyHash pool size `m` rather than using the fly's
      own fixed ratios untuned.
- [ ] **Phase 3 — write it up properly**, with a real Related Work section
      (DenseFly, BioHash, FlyNN, the existing `flyhash` PyPI package) and
      an honest account of the Phase 1 investigation, including the
      retracted non-negativity hypothesis — that's part of the story, not
      something to omit.
- [ ] **Phase 4 — arXiv preprint.**
- [ ] **Phase 5 — ACM XRDS or a workshop**, using the arXiv version as the base.
- [ ] Implement real multi-probe/banded LSH indexing on top of the sparse
      codes to get actual sublinear query time at scale (this is the
      piece needed before claiming a latency win over FAISS).
- [x] Wire `FlyMemoryStore` into an actual agent loop (`agent_memory.py` +
      `agent_demo.py`, Claude API tool-use) — demonstrated cross-turn
      recall of facts shared earlier in a conversation.
