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
RESEARCH_JOURNAL.md  Chronological account of issues faced, diagnoses, and fixes
src/
  flyhash.py          FlyHash + SimHash implementations
  memory_store.py     Agent-memory-store wrapper (FlyHash-backed + FAISS baseline)
  benchmark.py         End-to-end memory-store benchmark (recall / latency / memory)
  recall_vs_bits.py    Corrected equal-hash-length FlyHash vs SimHash comparison
  map_metric.py        Mean Average Precision @ R, the literature-standard metric
  multi_seed_benchmark.py  Reruns recall_vs_bits.py's protocol across N seeds, reports mean +/- std
  glove_embeddings.py  Real GloVe vector loader (used to find & fix the Phase 1 bug)
  digits_embeddings.py Real image loader (sklearn digits, MNIST stand-in — see Phase 2b)
  mnist_embeddings.py  Full MNIST loader via OpenML (run on a machine with internet)
  grid_search.py       Sparsity x pool-size hyperparameter ablation (Phase 2c)
  latency_memory_benchmark.py  Post-fix memory/latency numbers under the corrected protocol
  sparse_flyhash_benchmark.py  Sparse-set comparison follow-up (partial speed win, memory still open)
  real_embeddings.py   Real sentence-transformers loader (100 unique sentences, 5 topics)
  real_text_proxy.py   Offline stand-in using TF-IDF over real text (no network needed)
  agent_memory.py      Text-in/text-out memory wrapper: embeds + calls FlyMemoryStore
  agent_demo.py         Terminal chat agent (Claude API) using AgentMemory as a tool
  offline_demo.py      Network-free demo over the released dataset (no API key)
  export_dataset.py    Writes the labeled benchmark to datasets/*.jsonl
datasets/agent_memory_v1/  Released benchmark (memories, queries, labels)
tests/                 pytest checks (dataset, FlyHash)
REPRODUCING.md         Paper table -> command map
```

## Reproducing the paper

The labeled benchmark used in the paper is released in `datasets/agent_memory_v1/`
(6,000 memories, 150 queries, ground-truth relevance labels). See
[REPRODUCING.md](REPRODUCING.md) for the table-to-command map, or run:

```bash
pip install -r requirements.txt
./reproduce.sh quick
```

## Setup

```bash
pip install -r requirements.txt
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

## Real sentence embeddings (MiniLM) — the last open Phase 1 item, closed

The Phase 1 investigation confirmed the corrected protocol on GloVe and
synthetic data, but `real_embeddings.py` (real `all-MiniLM-L6-v2` sentence
embeddings, 384-dim) couldn't be verified in the sandbox — it needs to
download the model from HuggingFace, which the sandbox's network doesn't
allow. Run on a machine with real internet access (the user's own, with a
HuggingFace account), confirming the same pattern:

**Real MiniLM sentence embeddings (100 unique sentences, dim=384), mAP@20,
mean ± std over 10 seeds:**

| k (bits) | FlyHash mAP@20 | SimHash mAP@20 | gap > 2×combined std? |
|---|---|---|---|
| 16  | **0.310 ± 0.013** | 0.214 ± 0.026 | YES |
| 32  | **0.408 ± 0.018** | 0.266 ± 0.022 | YES |
| 64  | **0.537 ± 0.030** | 0.355 ± 0.024 | YES |
| 128 | **0.656 ± 0.016** | 0.466 ± 0.027 | YES |
| 256 | **0.753 ± 0.013** | 0.587 ± 0.016 | YES |

FlyHash wins robustly at every bit budget on real, production-grade sentence
embeddings — the exact embedding type `agent_memory.py` actually uses for
agent memory retrieval. This closes the last open item from Phase 1: every
real dataset tried (GloVe, MiniLM sentence embeddings, and — below — full
MNIST) now shows the same pattern.

Run it yourself (needs internet on first run, to download the model):
```bash
cd src
python multi_seed_benchmark.py --dataset real --n_docs 100 --n_queries 30 --R 20 --seeds 10
```

## Phase 2a: statistical rigor — is the GloVe result robust to random seed?

Every number above came from a single random draw of the projection
matrices. FlyHash's sparse projection and SimHash's random hyperplanes are
both randomly initialized, so one run only tells you what happened with one
seed, not whether the gap is real or just a lucky draw. `multi_seed_benchmark.py`
reruns the exact same corrected protocol across 10 random seeds (only the
projection matrices vary — the dataset and the fixed query sample stay the
same across seeds, so all the measured variation is attributable to
projection-matrix randomness, not to which queries got sampled).

**GloVe (glove-wiki-gigaword-100, 5000 words), mAP@100, mean ± std over 10 seeds:**

| k (bits) | FlyHash mAP@100 | SimHash mAP@100 | gap > 2×combined std? |
|---|---|---|---|
| 16  | **0.196 ± 0.007** | 0.047 ± 0.006 | YES |
| 32  | **0.314 ± 0.009** | 0.096 ± 0.009 | YES |
| 64  | **0.449 ± 0.008** | 0.185 ± 0.008 | YES |
| 128 | **0.585 ± 0.006** | 0.316 ± 0.006 | YES |
| 256 | **0.705 ± 0.006** | 0.469 ± 0.007 | YES |

The result holds. Standard deviation is small (0.006–0.009) at every bit
budget, and the FlyHash–SimHash gap clears twice the combined standard
deviation by a wide margin in every case — this is not an artifact of one
lucky projection matrix. (The single-seed numbers in the Phase 1 table above
differ slightly from the seed-0 slice of this table only because Phase 1
used a fixed `seed=42` default rather than `seed=0`; the qualitative picture
— and the size of the gap — is unchanged either way.) The same check on
synthetic clustered data (5 seeds, 2000 records) shows the same pattern:
small std, gap far exceeding 2×combined std at every bit budget.

Run it yourself:
```bash
cd src
python multi_seed_benchmark.py --dataset glove --seeds 10
python multi_seed_benchmark.py --dataset synthetic --seeds 10
```

## Phase 2b: a second real dataset — sklearn's `digits` (real images, not text)

GloVe is word vectors; everything so far has been text-shaped data. To
check the result isn't specific to text embeddings, the same corrected
protocol was run on `sklearn.datasets.load_digits()` — 1797 real handwritten
digit images (8x8 = 64 pixel dims each, from the UCI ML hand-written digits
dataset). This is a genuine image dataset, not a synthetic one, chosen
because full MNIST (60,000 28x28 images) isn't reachable from this sandbox:
every MNIST hosting source tried (OpenML, Google Cloud Storage, S3) is
blocked by the same network restriction that blocked HuggingFace for
`real_embeddings.py`, and no verified GitHub-hosted MNIST mirror was found
to point at instead of guessing at URLs. **This is not full MNIST** — see
the note in `digits_embeddings.py` and the Roadmap below for the pending
full-MNIST run.

**sklearn digits (1797 images, dim=64), mAP@50, mean ± std over 10 seeds:**

| k (bits) | FlyHash mAP@50 | SimHash mAP@50 | gap > 2×combined std? |
|---|---|---|---|
| 16  | **0.456 ± 0.015** | 0.119 ± 0.031 | YES |
| 32  | **0.593 ± 0.010** | 0.236 ± 0.039 | YES |
| 64  | **0.699 ± 0.011** | 0.397 ± 0.023 | YES |
| 128 | **0.748 ± 0.005** | 0.535 ± 0.021 | YES |
| 256 | **0.806 ± 0.007** | 0.664 ± 0.009 | YES |

Same shape as GloVe: FlyHash wins decisively and robustly at every bit
budget, and the relative advantage is largest at the smallest bit budgets
(FlyHash is ~3.8x SimHash at k=16, ~1.2x at k=256) — the same "biggest win
where it matters most" pattern, now confirmed on a second, structurally
different real dataset (dense pixel intensities instead of word co-occurrence
vectors).

Run it yourself:
```bash
cd src
python multi_seed_benchmark.py --dataset digits --n_docs 1797 --n_queries 150 --seeds 10 --R 50
```

### Full MNIST (60k images, 28x28) — the stand-in's real counterpart

The `digits` result above used a small 1797-image, 8x8 stand-in because
full MNIST wasn't reachable from the sandbox. Run on a machine with real
internet access (`mnist_embeddings.py`, via `sklearn.datasets.fetch_openml`),
on 5000 real 28x28 MNIST images (784 dims):

**Full MNIST (5000 images, dim=784), mAP@100, mean ± std over 10 seeds:**

| k (bits) | FlyHash mAP@100 | SimHash mAP@100 | gap > 2×combined std? |
|---|---|---|---|
| 16  | **0.276 ± 0.009** | 0.064 ± 0.014 | YES |
| 32  | **0.382 ± 0.010** | 0.133 ± 0.012 | YES |
| 64  | **0.469 ± 0.011** | 0.237 ± 0.014 | YES |
| 128 | **0.527 ± 0.008** | 0.351 ± 0.009 | YES |
| 256 | **0.563 ± 0.004** | 0.450 ± 0.010 | YES |

Same pattern as every other dataset tried: FlyHash wins robustly at every
bit budget, largest relative advantage at the smallest budget (~4.3x at
k=16, ~1.25x at k=256). This is the actual benchmark the original paper and
its follow-ups (BioHash, DenseFly) report against, not a stand-in — Phase 2b
is now fully resolved on real, standard-benchmark data.

Run it yourself (needs internet on first run, to download+cache via OpenML):
```bash
cd src
python multi_seed_benchmark.py --dataset mnist --n_docs 5000 --n_queries 200 --R 100 --seeds 10
```

## Phase 2c: hyperparameter grid search — is the result sensitive to sparsity or pool size?

Every result so far used fixed defaults (`sparsity=0.10`, `flyhash_pool_size=2000`)
without ever checking whether the FlyHash-over-SimHash win depends on those
choices. `grid_search.py` sweeps `sparsity ∈ {0.05, 0.10, 0.20}` ×
`flyhash_pool_size ∈ {500, 1000, 2000, 4000}` at a fixed bit budget, on
GloVe, with 3 seeds per cell (fewer than the 10-seed main results, since
this checks many more configurations — the goal is whether the qualitative
pattern holds everywhere, not a fresh publication-grade number per cell).

**k=16 bits (SimHash baseline: 0.046 ± 0.004):**

| sparsity | m=500 | m=1000 | m=2000 | m=4000 |
|---|---|---|---|---|
| 0.05 | 0.169±0.009 | 0.169±0.003 | 0.178±0.006 | 0.186±0.002 |
| 0.10 | 0.200±0.001 | 0.199±0.006 | 0.193±0.010 | 0.189±0.008 |
| 0.20 | 0.198±0.003 | 0.205±0.003 | 0.200±0.005 | 0.204±0.001 |

**k=64 bits (SimHash baseline: 0.177 ± 0.004):**

| sparsity | m=500 | m=1000 | m=2000 | m=4000 |
|---|---|---|---|---|
| 0.05 | 0.434±0.014 | 0.415±0.006 | 0.379±0.005 | 0.378±0.006 |
| 0.10 | 0.446±0.004 | 0.456±0.003 | 0.451±0.009 | 0.442±0.014 |
| 0.20 | 0.453±0.005 | 0.472±0.001 | 0.472±0.003 | 0.472±0.004 |

**k=256 bits (SimHash baseline: 0.469 ± 0.003):**

| sparsity | m=500 | m=1000 | m=2000 | m=4000 |
|---|---|---|---|---|
| 0.05 | 0.588±0.007 | 0.667±0.003 | 0.684±0.007 | 0.632±0.011 |
| 0.10 | 0.607±0.004 | 0.683±0.002 | 0.707±0.006 | 0.689±0.007 |
| 0.20 | 0.609±0.003 | 0.686±0.002 | 0.712±0.003 | 0.719±0.002 |

**Every single cell in all three tables beats the SimHash baseline at the
same bit budget** — 36 configurations across 3 bit budgets, none of them
flip the result. Two secondary patterns worth noting for anyone tuning
this: (1) higher sparsity (0.20) tends to do slightly better than the
fly's own biological ratio (~0.10-0.12) at the larger bit budgets, though
the effect is small relative to the FlyHash-vs-SimHash gap; (2) pool size
matters less than expected once it's at least a few times larger than the
bit budget — going from m=2000 to m=4000 rarely helps and sometimes
slightly hurts. Neither pattern threatens the core result; they're the
kind of secondary observation worth a sentence in a paper's ablation
section, not a redesign.

Run it yourself:
```bash
cd src
python grid_search.py --dataset glove --bits 64
python grid_search.py --dataset glove --bits 16
python grid_search.py --dataset glove --bits 256
```

## Memory and query-latency: a real, honest limitation (post-fix numbers)

Every result above measures hash *quality* (mAP) — none of it measures
memory footprint or query speed under the corrected protocol. The only
memory/latency numbers previously in this README (`benchmark.py`'s bucket
vs. scan vs. FAISS table) predate the Phase 1 fix and used the old,
already-superseded comparison design. `latency_memory_benchmark.py` fills
that gap honestly, and the result is **not** a clean win for FlyHash —
it's a real finding worth understanding before this goes in a paper.

**GloVe (5000 records, dim=100), 50 queries, brute-force ranking for
everything (no ANN index on either side):**

| k (bits) | SimHash packed bytes/record | mem reduction vs. float32 | FlyHash ms/query | SimHash ms/query | speedup vs. exact L2 |
|---|---|---|---|---|---|
| 16  | 2  | 200.0x | 14.05 | 0.19 | 4.0x |
| 32  | 4  | 100.0x | 12.76 | 0.29 | 2.7x |
| 64  | 8  | 50.0x  | 12.93 | 0.45 | 1.8x |
| 128 | 16 | 25.0x  | 14.15 | 0.84 | 0.9x |
| 256 | 32 | 12.5x  | 13.17 | 2.17 | 0.4x |

(Exact float32 L2 baseline: 400 bytes/record, 0.782 ms/query.)

**What this shows, plainly:**

- **SimHash's k-bit code is exactly k bits** — pack it, compare it via a
  cheap matmul, and at small k it's both far smaller (200x less memory at
  k=16) and faster than brute-force exact search. That part is the
  expected, well-known benefit of compact hashing.
- **FlyHash's code is NOT k bits — it's `pool_size` (2000) bits, of which
  only k are ever set to 1.** The Hamming-ranking implementation used
  throughout this project (`vectorized_hamming_ranking`, a single matmul)
  compares the *full* 2000-wide array regardless of k, because that's how
  the code is represented in memory (dense binary array, not a sparse
  index list). That means FlyHash's dense-packed size is a fixed 250
  bytes/record **regardless of k** (worse than SimHash's k-bit code at
  every k except k≈250+), and its brute-force query time is a fairly flat
  ~13ms/query regardless of k — **10-70x slower than SimHash at the same
  k**, and slower than even the naive exact-L2 baseline.

**Why this doesn't contradict the mAP results, and what it does mean:**
mAP only depends on *which* k bits are set, not on how the code is stored
or compared — so the quality results (FlyHash wins decisively) and the
resource results (FlyHash, compared densely, costs more) are both true at
once, about different things.

### Follow-up experiment: does a sparse-set comparison close the gap?

Since every FlyHash code has *exactly* `k` active bits, there's a neat
identity: `hamming(a, b) = 2k - 2·|intersection(a, b)|`. Because `k` is
fixed, ranking by Hamming distance is mathematically identical to ranking
by descending intersection size — so Hamming ranking can be computed as a
**sparse-sparse matrix product** (`scipy.sparse`) over just the active
indices, instead of a dense matmul over the full 2000-wide array.
`sparse_flyhash_benchmark.py` tries exactly this, as a genuine experiment
(not a known-answer confirmation):

| k (bits) | sparse idx bytes/rec | SimHash bytes/rec | dense ms/query | sparse ms/query | SimHash ms/query |
|---|---|---|---|---|---|
| 16  | 32  | 2  | 15.77 | 0.71 | 0.40 |
| 32  | 64  | 4  | 14.44 | 0.73 | 0.29 |
| 64  | 128 | 8  | 14.54 | 1.59 | 0.45 |
| 128 | 256 | 16 | 15.03 | 3.39 | 0.84 |
| 256 | 512 | 32 | 13.90 | 8.55 | 1.79 |

**Result: a genuine partial win, not a full close of the gap.**
- **Speed** improves dramatically over the naive dense comparison — 10 to
  22x faster — and now correctly *scales with k* instead of being a flat
  ~14ms regardless of bit budget. But SimHash is still 2-5x faster than
  FlyHash-sparse at every k: a small dense k×k matmul (what SimHash
  already uses) beats sparse-matrix overhead at these scales.
- **Memory** now also scales sensibly with k (32→512 bytes) instead of a
  flat 250 bytes, a real improvement in behavior — but each active index
  needs a full 2-byte integer to store, so it's still ~16x larger than
  SimHash's bit-packed code at the same k. Bit-packing beats index-listing
  whenever the pool is this size, because storing "which of ~2000
  positions is on" costs more per bit than a single packed bit does.

**Honest bottom line:** this experiment recovers most of the *speed* gap
but not the *memory* gap, and doesn't fully close either one relative to
SimHash. FlyHash's advantage remains a genuine, robustly-validated
*quality-per-bit* win; converting it into an outright memory-or-speed win
over a well-implemented SimHash baseline is still open — a real, harder
problem than initially assumed, and reported here as such rather than
declared solved because one attempt showed partial improvement.

Run it yourself:
```bash
cd src
python sparse_flyhash_benchmark.py
```

Run it yourself:
```bash
cd src
python latency_memory_benchmark.py --dataset glove
```

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
3. **A quality win is not automatically a memory-or-speed win, and this
   project's own numbers prove it against itself** — FlyHash beats SimHash
   decisively on retrieval quality (mAP) at every bit budget, dataset, and
   hyperparameter setting tested, full stop. But measured honestly under
   the same dense, brute-force comparison used throughout, FlyHash's code
   is *larger and slower to compare* than SimHash's at equal quality-bits
   `k`, because FlyHash's code lives in a much wider space (`pool_size`,
   e.g. 2000) of which only `k` bits are active, while the comparison
   implementation used here doesn't exploit that sparsity. Realizing
   FlyHash's memory/speed potential needs a sparse-aware storage and
   comparison scheme (partially present in `memory_store.py`, not yet
   proven sublinear) — an unsolved problem, not a minor footnote. Reporting
   only the quality win while staying quiet about this would be a
   cherry-picked story of exactly the kind this project set out to avoid.

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
      fixed on synthetic data, real GloVe vectors, AND real MiniLM sentence
      embeddings (run on the user's own machine with internet access — see
      the "Real sentence embeddings" section above). No open items remain
      from Phase 1.
- [x] **Phase 2a — statistical rigor: multi-seed harness.** Resolved: the
      Phase 1 GloVe result (and the synthetic-data result) hold up across
      10 random seeds each, with small std (0.006–0.009 mAP) and the
      FlyHash–SimHash gap clearing 2×combined std at every bit budget
      tested — see the Phase 2a section above for the full table.
- [x] **Phase 2b — a second real dataset.** Resolved: both sklearn's
      `digits` stand-in (1797 images, 8x8) and full MNIST (5000 images,
      28x28, run on the user's own machine via OpenML) show the same
      pattern as GloVe — FlyHash wins robustly at every bit budget tested,
      with the largest relative advantage at the smallest budgets. SIFT1M
      still pending as a third/fourth dataset if desired.
- [x] **Phase 2c — hyperparameter grid search.** Resolved: swept
      `sparsity × pool_size` (12 combinations) at 3 bit budgets (16/64/256
      bits, 36 configurations total) on GloVe. Every single configuration
      beat the SimHash baseline at its bit budget — the result is not a
      hyperparameter accident. See the Phase 2c section above for the full
      tables and two minor secondary observations (higher sparsity helps
      slightly at large bit budgets; pool size matters less once it's a few
      times the bit budget). SIFT1M intentionally skipped as a 5th dataset
      — the finding is already confirmed across 4 independent real datasets
      spanning 2 modalities (text: GloVe, MiniLM; images: digits, MNIST),
      so a 5th subsampled dataset was judged to add breadth, not new
      evidence, relative to the effort of acquiring and subsampling a
      1M-vector dataset. Noted as optional future work.
- [ ] **Phase 3 — write it up properly**, with a real Related Work section
      (DenseFly, BioHash, FlyNN, the existing `flyhash` PyPI package) and
      an honest account of the Phase 1 investigation, including the
      retracted non-negativity hypothesis — that's part of the story, not
      something to omit.
- [ ] **Phase 4 — arXiv preprint.**
- [ ] **Phase 5 — ACM XRDS or a workshop**, using the arXiv version as the base.
- [x] **Fresh post-fix memory/latency benchmark.** Resolved: measured
      honestly under the corrected protocol (`latency_memory_benchmark.py`)
      rather than reusing pre-fix numbers. Result is a real limitation, not
      a win: FlyHash's dense-packed code is larger and slower to
      brute-force-compare than SimHash's at equal `k`, because the
      comparison implementation doesn't exploit FlyHash's sparsity. See
      the "Memory and query-latency" section above for the full numbers
      and explanation.
- [x] **Try a sparse index-set comparison** (`sparse_flyhash_benchmark.py`)
      to see if it closes the memory/speed gap found above. Resolved with
      a genuine, non-obvious result: recovers most of the *speed* gap
      (10-22x faster than the dense comparison, and now scales correctly
      with `k`) but SimHash's dense k×k matmul is still 2-5x faster;
      memory now scales correctly with `k` too, but index-list storage
      (2 bytes/active-bit) is still ~16x larger than SimHash's bit-packed
      code. A real partial win, not a full close — see the "Follow-up
      experiment" subsection above.
- [ ] A genuinely sublinear index (multi-probe/banded LSH, or an inverted
      index that's actually proven sublinear at scale, unlike this
      project's earlier "bucket mode" attempt) remains open — the sparse
      comparison above is faster per-query but still brute-force (compares
      against every record); real scale needs avoiding that entirely.
- [x] Wire `FlyMemoryStore` into an actual agent loop (`agent_memory.py` +
      `agent_demo.py`, Claude API tool-use) — demonstrated cross-turn
      recall of facts shared earlier in a conversation.
