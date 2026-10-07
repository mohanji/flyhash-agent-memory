# Research Journal — FlyHash for LLM Agent Memory

This file is a chronological account of the project: what was tried, what
broke, how it was diagnosed, and what was learned. It is kept alongside
the code deliberately — the dead ends and wrong turns are as much a part
of the research record as the final numbers, and a reader (including
future-me) should be able to reconstruct *why* the code looks the way it
does, not just what it currently does.

Entries are grouped by stage rather than by date, since stages sometimes
spanned multiple sessions.

---

## Stage 1 — Core algorithm implementation

**Goal:** reimplement FlyHash (Dasgupta, Stevens & Navlakha, 2017) from
the paper's description rather than using an existing package, plus a
standard SimHash baseline to compare against.

**What was built:** `flyhash.py` (sparse random projection + k-winners-
take-all), `memory_store.py` (a swappable agent-memory-store interface
wrapping either FlyHash or FAISS).

**Issue — sandbox network restrictions.** The development sandbox's
network is locked to a specific domain allowlist. `huggingface.co` (needed
for `sentence-transformers`) is not on it, so real sentence embeddings
could not be downloaded or tested directly in that environment.
**Resolution:** built `real_embeddings.py` (the real loader, meant to be
run on a machine with normal internet access) alongside
`real_text_proxy.py`, an offline stand-in using TF-IDF over a small
hand-written corpus, so the pipeline logic could be validated without
network access.

**Issue — disk space.** An early `pip install sentence-transformers`
attempt failed with `[Errno 28] No space left on device`. **Resolution:**
`pip cache purge` freed enough space; later, GloVe/gensim installs were
handled the same way. Lesson: check disk headroom before installing
anything that pulls in PyTorch.

---

## Stage 2 — First benchmark, and a premature framing

**What was built:** `benchmark.py`, an end-to-end memory-store benchmark
(recall / latency / memory vs. a FAISS exact-search baseline), initially
on synthetic clustered data (`sklearn.datasets.make_blobs`, chosen because
the intended real dataset, 20 Newsgroups, requires a network download this
sandbox couldn't make).

**Issue — a "bucket" (inverted-index) retrieval mode was added to
`FlyMemoryStore` expecting sublinear query time, but it wasn't actually
sublinear.** With 20,000 records and only 60 semantic clusters, ~75% of
the store shared at least one active hash cell with any given query, so
the candidate set barely shrank versus a full scan. **Resolution:** kept
the "scan" (exact Hamming ranking) mode as the correct, honestly-labeled
option; documented that real sublinear retrieval from sparse codes needs
banding/multi-probe LSH, which was not implemented at this stage.

**Issue — the FAISS-vs-FlyHash comparison in `benchmark.py` was not a
fair comparison of hash quality.** FAISS was searching the full,
uncompressed embeddings, while FlyHash was working from a compressed
code — comparing compression to no-compression rather than isolating the
hash function itself. **Resolution:** built `recall_vs_bits.py`, holding
the bit budget equal between FlyHash and SimHash so the comparison
isolates hash-function quality from how much either one was compressed.

**A wrong turn, not yet visible as wrong:** on the isotropic synthetic
data, SimHash beat FlyHash at every bit budget tested. This was
interpreted as an expected, explainable result: isotropic data has no
redundancy for FlyHash's sparse projection to exploit. This interpretation
was reasonable given the evidence at the time, but — as Stage 5 later
showed — it was actually caused by a flawed comparison protocol, not a
real property of the data.

---

## Stage 3 — Real text, and a hypothesis that would later be retracted

**What was built:** ran `recall_vs_bits.py` with `real_text_proxy.py`
(TF-IDF over real sentences) and, once the user ran it locally with
internet access, with `real_embeddings.py` (`sentence-transformers`,
`all-MiniLM-L6-v2`).

**Result at the time:** FlyHash beat SimHash on TF-IDF (non-negative,
correlated features), but lost badly on real MiniLM sentence embeddings
(dense, signed, decorrelated by contrastive training).

**Hypothesis formed (later retracted):** FlyHash's advantage depends on
non-negative, redundant input structure — matching the fly's own
non-negative firing-rate biology — and doesn't transfer to dense signed
embeddings. This was written up in the README as a finding, with
reasoning that sounded plausible: the k-winners-take-all step "discards
useful signal" in a decorrelated signed regime rather than "discarding
redundancy."

**Why this was dangerous, in hindsight:** the hypothesis was internally
consistent and explained the observed data, which made it easy to accept
without independently verifying it against the literature. It was
eventually falsified in Stage 5 — not by new data, but by discovering that
the *comparison itself* was broken, which had been silently producing an
apparent effect that had nothing to do with non-negativity at all.

---

## Stage 4 — Local environment setup (Windows)

This stage was mostly tooling friction rather than research, but is
recorded since it consumed real time and has generalizable lessons.

**Issue — PyTorch install failed with a Windows long-path error.** The
project's original folder path was long and contained a space
(`C:\Mohan\AI\AI_APPS\FlyHash\FlyHash Bio-Inspired Sparse Hashing for LLM
Agent Memory\flyhash-agent-memory\flyhash-agent-memory\venv\...`), and a
zip-extraction quirk had also nested the folder inside itself.
**Resolution:** moved the project to a short path (`C:\flyhash`) and
rebuilt the virtual environment there, rather than enabling Windows'
registry-level long-path support (a valid alternative, but a bigger
change for a problem a simple move avoided).

**Issue — `source venv/Scripts/activate` failed.** `source` is a bash
builtin; it doesn't exist in PowerShell or CMD. **Resolution:**
`venv\Scripts\Activate.ps1` in PowerShell, `venv\Scripts\activate.bat` in
CMD — and a reminder that PowerShell-only commands (`Remove-Item`) fail
silently confusingly when run in a CMD session by mistake, which happened
once here too (the prompt without a `PS` prefix was the tell).

**Issue — dimension mismatch in `recall_vs_bits.py` when swapping in real
embeddings.** The function trusted a passed-in `embed_dim` parameter to
construct `FlyHash`/`SimHash`, but `real_embeddings.py`'s loader ignores
that parameter (MiniLM always outputs 384-dim vectors regardless of what's
requested), causing a matmul shape error. **Resolution:** read the actual
embedding dimensionality off the returned array (`embeddings.shape[1]`)
instead of trusting the requested parameter.

**Issue — degenerate recall numbers from a duplicated corpus.**
`real_embeddings.py`'s `SAMPLE_CORPUS` originally had only 25 unique
sentences, repeated to fill `n_docs=5000`. Repeated sentences produce
identical embeddings, creating hundreds of exact-distance-zero "ties" per
query; which specific ties land in a "top-10" becomes arbitrary
tie-breaking noise rather than a measure of hash quality.
**Resolution:** expanded the corpus to 100 genuinely unique sentences and
documented the failure mode directly in the file so it isn't repeated.

---

## Stage 5 — The GloVe discrepancy: found, investigated, and resolved

**The trigger.** Before publishing anything, the plan called for checking
this project's finding against the literature. The original FlyHash paper
tested on GloVe embeddings — dense and signed, the same category as
MiniLM — and reported FlyHash *winning* there. That directly contradicted
the Stage 3 finding and the non-negativity hypothesis built on it.

**Step 1 — ruled out normalization.** L2-normalizing GloVe vectors before
hashing made no meaningful difference to the outcome.

**Step 2 — ruled out signed vs. true binary projection weights.** The
paper defines the sparse projection matrix as binary {0,1}; the
implementation had used random ±1 weights. Switching to true binary
weights made no meaningful difference either.

**Step 3 — ruled out the evaluation metric.** Recall@k (simple top-k
overlap) was swapped for mean Average Precision @ R (mAP@R), confirmed via
the BioHash paper (Ryali et al., 2020) as the metric this literature
actually standardizes on. Still no meaningful change — SimHash still won
clearly.

**Step 4 — found it, by diffing against a reference implementation.**
Cloned the official reference code (`github.com/dataplayer12/Fly-LSH`) and
read its `flylsh` class directly, rather than continuing to guess. The
key discovery: the paper's fair comparison holds the number of **winning
bits** (`hash_length`, called `k` in this repo) equal between FlyHash and
conventional LSH — NOT the total hash dimensionality. FlyHash's
`embedding_size` (the Kenyon-cell pool that winners are chosen from,
called `m`) is a separate, much larger, independent parameter.

Every benchmark run up to this point had swept `hash_dim` directly while
holding `wta_percent` fixed at 5%. That silently varied FlyHash's
*effective* code length (`k`, the number of active bits) to a small
fraction of whatever nominal "bits" value SimHash was being given at the
same setting — for example, at `hash_dim=128`, FlyHash's `k` was only
about 6 active bits, compared head-to-head against SimHash's full 128
dense bits. That is not a fair "equal bit budget" comparison; it
systematically disadvantaged FlyHash in every prior result.

**The fix.** `recall_vs_bits.py` was rewritten to hold `k` (the swept
"bits" parameter) equal for both methods, with FlyHash's pool size `m`
fixed separately and large (e.g. 2000), matching the reference
implementation's actual parameterization.

**The corrected result, real GloVe vectors (5000 words, mAP@100):**

| k (bits) | FlyHash mAP@100 | SimHash mAP@100 |
|---|---|---|
| 16  | 0.215 | 0.043 |
| 64  | 0.458 | 0.195 |
| 256 | 0.707 | 0.480 |

FlyHash now wins decisively at every bit budget, matching the original
paper's pattern. Re-running the corrected protocol on the Stage 2
synthetic isotropic data showed FlyHash winning there too — meaning the
Stage 3 non-negativity hypothesis, which had explained a real and
reproducible pattern in the data, was explaining an artifact of a broken
comparison, not a genuine property of either dataset or algorithm.

**Retraction, stated plainly:** the claim "FlyHash's advantage depends on
non-negative, redundant input structure" is **not supported**. It was a
plausible-sounding explanation for a gap that had a different, mundane
cause. It is left in this journal (and referenced, not asserted, in the
README) as a record of the investigation, not as a finding to build on.

---

## Stage 6 — Agent integration and tooling friction

**What was built:** `agent_memory.py` (a text-in/text-out wrapper around
`FlyMemoryStore`) and `agent_demo.py` (a terminal chat agent using the
Claude API with `save_memory`/`recall_memory` tools), to demonstrate the
hashing work actually being useful inside an agent loop rather than only
existing as a benchmark.

**Issue — Anthropic Console signup transient error.** A generic "Server
Error" appeared when first loading `console.anthropic.com`. Checked
`status.anthropic.com` (no active incident reported) and resolved with a
hard refresh / different browser — a session-local issue, not a service
outage.

**Issue — copied the wrong credential into the API key environment
variable.** The value set via `setx` started with `apikey_...`, not the
`sk-ant-api03-...` prefix real Anthropic keys use — almost certainly a
masked/placeholder value copied from the console's key list view rather
than the one-time full-key reveal shown immediately after clicking
"Create Key." **Resolution:** generated a fresh key and copied it from the
one-time reveal dialog directly.

**Issue — `AuthenticationError: invalid x-api-key` even after resetting
the key.** Diagnosed with a minimal one-line API call outside the full
agent script, isolating the problem to the key itself rather than
`agent_demo.py`'s code.

**Issue — `BadRequestError: credit balance too low`, after authentication
finally succeeded.** This clarified that Claude.ai access and API credits
are separate, prepaid systems — a distinction worth stating plainly since
it's a common point of confusion. Resolved by adding credits via
Plans & Billing in the Console.

---

## Stage 7 — Publication planning

Before committing to a publication push, the plan itself was pressure-
tested rather than assumed: a literature search surfaced a denser
follow-up ecosystem than initially assumed (DenseFly, BioHash, FlyNN, an
existing `flyhash` PyPI package), which meant a plain reproduction would
have limited novelty on its own. The most defensible novel angle
identified was the LLM-agent-memory application, not the hashing
algorithm itself — no prior work applying FlyHash specifically to agent
memory retrieval was found. A five-phase plan was set: resolve the GloVe
discrepancy properly (Stage 5, now done), add real benchmark datasets and
statistical rigor, write up with a proper related-work section, publish to
arXiv, then target ACM XRDS or a workshop using the arXiv version as the
base.

---

## Stage 8: Phase 2a — is the GloVe result robust, or a one-seed fluke?

**Issue.** Every number reported through Stage 7 — including the corrected
GloVe result that resolved the Phase 1 discrepancy — came from a single
random draw of the projection matrices (a fixed `seed=42` default). Both
FlyHash's sparse projection and SimHash's random hyperplanes are randomly
initialized, so a single run only shows what happened with one particular
seed. Before writing this up for publication, or building further phases on
top of it, the honest question was: does the FlyHash-over-SimHash gap
survive across many independent random draws, or could it have been an
artifact of one favorable seed?

**Investigation.** Built `multi_seed_benchmark.py`, which reruns the exact
corrected comparison protocol from `recall_vs_bits.py` across N seeds. Only
the projection-matrix randomness is varied per seed; the dataset and the
query sample are both held fixed across all seeds, so any measured
variation is attributable purely to projection-matrix noise rather than to
which queries happened to get sampled. For each seed and each bit budget,
both FlyHash's mAP@100 and SimHash's mAP@100 are recorded, then reported as
mean ± std, plus a simple robustness check: does the FlyHash–SimHash gap
exceed 2× the combined standard deviation of the two methods?

Hit one environment snag before getting a result: the harness was first
written and validated in a container session, but the actual full run
happened in a fresh session where none of the project's dependencies
(gensim, etc.) were pre-installed and the rest of the project tree wasn't
present yet — a reminder that "it ran once" and "it's reproducible from a
clean checkout" are different claims. Reinstalled dependencies, restored
the project from the last shipped copy, and reran from a clean state before
trusting the numbers.

**Resolution.** Ran 10 seeds on real GloVe (5000 words) and 5 seeds on the
synthetic clustered dataset. Results:

- GloVe, mAP@100: std ranged 0.006–0.009 across all five bit budgets
  (k=16,32,64,128,256); FlyHash beat SimHash by a wide margin at every k,
  and the gap cleared 2×combined-std at every k. Example: k=64, FlyHash
  0.449 ± 0.008 vs. SimHash 0.185 ± 0.008 — a gap of 0.264 against a
  combined std of roughly 0.011.
- Synthetic clustered data: same pattern — small std (0.003–0.014), gap
  clears 2×combined-std at every bit budget.

**Conclusion.** The Phase 1 finding is not a one-seed artifact. This is the
first result in the project backed by a variance estimate rather than a
single point number, which matters for the eventual write-up: reviewers of
a paper making an empirical claim will expect exactly this kind of check,
and now it exists and is reproducible with one command
(`python multi_seed_benchmark.py --dataset glove --seeds 10`).

---

## Stage 9: Phase 2b — does the result hold on a second, non-text dataset?

**Issue.** Every real-data result so far (GloVe, and the earlier MiniLM/TF-IDF
work) was text-derived. Before treating the finding as general, it needed
checking on data from a genuinely different modality — image pixels rather
than word co-occurrence statistics. Full MNIST was the natural first choice,
matching what the original FlyHash paper and most follow-up papers
(BioHash, DenseFly) actually benchmark against.

**Investigation.** Tried to fetch full MNIST (60,000 28x28 images) from
inside the sandbox: `sklearn.datasets.fetch_openml("mnist_784")` failed
(OpenML's API blocked — `403 Forbidden` through the sandbox's network
proxy). Tried the two other common MNIST hosting sources directly —
`storage.googleapis.com` (what Keras's loader uses) and
`ossci-datasets.s3.amazonaws.com` (what torchvision's loader uses) — both
also blocked. Confirmed the sandbox's network allowlist really is narrow:
`raw.githubusercontent.com` is reachable (same as the earlier GloVe-via-
gensim download, which used a GitHub-backed mirror), but a GitHub API code
search for a hosted MNIST CSV came back `403` too (GitHub's code-search
endpoint requires authentication), so there was no way to locate a mirror
to point at without guessing at unverified URLs — decided against guessing,
since a URL invented from memory could easily 404 or point at the wrong
file. This is the same category of limitation as `real_embeddings.py`
hitting the HuggingFace block in Stage 3/4.

**Resolution.** Used `sklearn.datasets.load_digits()` instead — ships
inside scikit-learn itself (no network call needed at all), a genuine real
image dataset (1797 handwritten digit images, 8x8=64 pixels, from the UCI
ML repository), just smaller and lower-resolution than full MNIST. Built
`digits_embeddings.py` with the same `load_embeddings(...)` interface as
every other loader, and explicitly documented in its docstring and in the
README that this is *not* full MNIST, with the exact swap needed
(`fetch_openml` or `torchvision.datasets.MNIST`) flagged as a pending
item for a machine with real internet access — same honesty pattern as
the still-open real-MiniLM verification from Phase 1.

Ran the full 10-seed harness on it: FlyHash beat SimHash robustly (gap
clears 2×combined-std) at every bit budget from 16 to 256, with the same
"biggest relative win at the smallest bit budget" shape seen on GloVe
(~3.8x at k=16, ~1.2x at k=256). This is now confirmed on two structurally
different real modalities (word vectors, image pixels), not just one.

**What this does and doesn't establish.** It's evidence the pattern isn't
an artifact specific to word-embedding geometry. It is *not* the same as
having the full MNIST number the original paper and its follow-ups report,
which stays an open item — a reviewer comparing directly against those
papers' tables will want the real 60k/28x28 result, not the 1797/8x8 stand-in.

---

## Stage 10: closing the sandbox gaps — real MiniLM and full MNIST, run on the user's own machine

**Issue.** Two open items had been sitting in the docs since Phase 1/2b,
both caused by the same root constraint: the sandbox environment this
project has mostly been developed in has a narrow network allowlist (GitHub
and package registries only), which blocked (1) verifying the corrected
protocol against real `all-MiniLM-L6-v2` sentence embeddings
(`real_embeddings.py` needs to hit huggingface.co), and (2) full MNIST
(every hosting source tried — OpenML, Google Cloud Storage, S3 — was
blocked; `digits_embeddings.py`'s sklearn `digits` dataset was used as an
honestly-labeled stand-in). Both were explicitly flagged as "needs a
machine with internet access" rather than worked around.

**Investigation/resolution.** Rather than keep guessing at sandbox
workarounds, the two loaders were wired into `multi_seed_benchmark.py`'s
dataset dispatcher (`--dataset real` for MiniLM, `--dataset mnist` for full
MNIST via a new `mnist_embeddings.py` using `sklearn.datasets.fetch_openml`),
and the user ran both directly on their own machine, which has real
internet access and a HuggingFace account. Two commands, no sandbox
involved:

```
python multi_seed_benchmark.py --dataset real --n_docs 100 --n_queries 30 --R 20 --seeds 10
python multi_seed_benchmark.py --dataset mnist --n_docs 5000 --n_queries 200 --R 100 --seeds 10
```

**Results.**

- **Real MiniLM sentence embeddings** (100 unique sentences, dim=384,
  mAP@20, 10 seeds): FlyHash beat SimHash robustly at every bit budget
  (e.g. k=64: 0.537±0.030 vs. 0.355±0.024). This is the exact embedding
  model `agent_memory.py` uses in the actual agent-memory demo, so this
  result directly validates the piece of the project that matters most for
  the "does this help a real LLM agent" claim, not just an abstract
  benchmark dataset.
- **Full MNIST** (5000 images, dim=784, mAP@100, 10 seeds): same pattern
  again (e.g. k=64: 0.469±0.011 vs. 0.237±0.014), and this is the literal
  benchmark the original paper and BioHash/DenseFly report against — no
  longer a stand-in.

**What this closes.** Every real dataset now attempted (GloVe, MiniLM
sentence embeddings, sklearn digits, full MNIST) — four independent,
structurally different sources of real data across two modalities (text,
images) — shows the identical qualitative pattern: FlyHash beats SimHash
robustly at every bit budget, with the relative advantage largest exactly
where bit budgets are smallest. At this point the finding has been checked
enough different ways that the honest conclusion is: this is very unlikely
to be a fluke of any one dataset, sandbox limitation, or evaluation choice.
Remaining Phase 2 items are optional breadth (SIFT1M as a fifth dataset,
hyperparameter grid search), not core validation — Phase 3 (write-up) is
now a reasonable next step whenever ready.

---

## Stage 11: Phase 2c — hyperparameter grid search, and deciding to skip SIFT1M

**Issue.** Every result up to this point used the same fixed defaults
(`sparsity=0.10`, `flyhash_pool_size=2000`) without ever testing whether the
FlyHash-over-SimHash result depended on those specific choices. Before
Phase 3 (write-up), this was the last real question standing between "this
result holds for the settings I happened to pick" and "this result holds
across the reasonable range of settings" — exactly the question a paper
reviewer would ask first.

There was also a live decision to make about SIFT1M: the original paper's
benchmark suite is roughly SIFT + GloVe + MNIST, so adding SIFT1M would
complete an exact-replication set. Weighed against that: SIFT1M is ~1M
128-dim vectors (~500MB, not GitHub-hosted, so likely another sandbox-block
situation requiring the user's own machine again), and to keep it tractable
it would need subsampling anyway — the same treatment already given to
GloVe and MNIST (5000-record samples). A subsampled 5th dataset, on top of
four already-confirmed real, structurally different datasets across two
modalities (text: GloVe, MiniLM; images: digits, MNIST), was judged to add
breadth without adding meaningfully to the evidence. Decided to skip it and
note it as explicit future work rather than chase a fifth confirmation of a
pattern that four independent datasets had already established.

**Investigation.** Built `grid_search.py`, which fixes a bit budget `k` and
sweeps `sparsity ∈ {0.05, 0.10, 0.20}` × `pool_size ∈ {500, 1000, 2000, 4000}`
(12 combinations), reporting FlyHash's mAP@100 (mean ± std over 3 seeds —
fewer than the 10-seed main harness, deliberately, since this sweeps many
more configurations and the goal is checking whether the qualitative
pattern survives everywhere, not producing a fresh publication-grade number
per cell) against the fixed SimHash baseline at the same k. Ran this at
three bit budgets (16, 64, 256) to check the low, middle, and high end of
the range already reported — 36 configurations total.

**Resolution.** All 36 configurations beat the SimHash baseline at their
respective bit budget. No combination of sparsity and pool size flipped the
result. Two secondary patterns worth a note (not a redesign): higher
sparsity (0.20, above the fly's own biological ~0.10-0.12 ratio) does
slightly better at larger bit budgets, and pool size stops mattering much
once it's a few times larger than the bit budget — the jump from m=2000 to
m=4000 rarely helps and occasionally hurts slightly.

**Conclusion.** The result isn't a hyperparameter accident. Combined with
the multi-seed statistical check (Stage 8) and the four-dataset replication
(Stages 5, 9, 10), this is now checked along every axis a reviewer would
reasonably ask about before Phase 3 (write-up): random seed, dataset,
modality, and hyperparameter choice. SIFT1M and further hyperparameter
tuning remain as explicit, deliberately-deferred future work — not gaps
in the current claim, but places to extend it later.

---

## Stage 12: honest memory/latency numbers, and an uncomfortable finding

**Issue.** While outlining the Phase 3 paper, it became clear the only
memory/latency numbers anywhere in this project (`benchmark.py`'s
bucket/scan/FAISS table) predate the Phase 1 fix and use the old,
already-superseded comparison design. A paper reporting the corrected mAP
results couldn't honestly cite pre-fix resource numbers alongside them —
they measure a configuration that's already been shown to be flawed.

**Investigation.** Built `latency_memory_benchmark.py` to measure, under
the exact corrected protocol used everywhere else in Phase 2 (fixed `k`,
large fixed pool `m`, real GloVe data): packed memory per record and
brute-force query latency, for both FlyHash and SimHash at equal `k`,
against an exact-float32-L2 baseline. Expected this to mostly confirm
FlyHash's practical benefits alongside its already-proven quality
advantage. It did not.

Digging into why: `vectorized_hamming_ranking` (used throughout this
project for all Hamming-distance ranking, including every mAP result
reported so far) does a single matmul over the full code width. For
SimHash, code width = k (e.g. 16, 64, 256). For FlyHash, code width =
`pool_size` (always 2000 in this project, e.g. via `flyhash_pool_size`),
regardless of k — only k of those 2000 bits are ever set to 1, but the
implementation stores and compares the full dense 2000-wide array. That
means FlyHash's packed memory is a flat ~250 bytes/record no matter what
k is (worse than SimHash's k-bit code at every k up to ~250), and its
brute-force query time is a roughly flat ~13ms/query regardless of k —
10-70x slower than SimHash at the same k, and slower than the naive
exact-L2 baseline entirely.

**Resolution — and the honest framing that matters here.** This is not a
bug to fix quietly; it's a real, structural fact about the current
implementation that needed reporting alongside the quality results, not
instead of investigating further before writing anything. Key points
worked out:
- mAP (quality) only depends on *which* k bits are set, so it's completely
  unaffected by how the code is stored/compared — the Phase 1/2 quality
  results all stand exactly as reported.
- Memory *can* be recovered by storing only the k active indices (a sparse
  index-list) instead of the dense 2000-bit array — `memory_store.py`
  already does this for actual storage, so real deployed memory is much
  closer to SimHash's footprint than this benchmark's dense-packing number
  suggests.
- Speed is the harder, still-unsolved half: comparing sparse index sets
  efficiently needs either a real sublinear index (the "bucket" mode in
  `memory_store.py` was already found, back in Stage 2, to not be
  genuinely sublinear at tested scale) or a fundamentally different
  comparison approach (set intersection instead of dense Hamming
  distance) — neither has been built or demonstrated in this project.

**Why this stays in, not gets buried.** The project's whole premise has
been "honest science over a clean demo." A quality win is genuinely
valuable and fully real — but claiming or implying a memory/speed win
alongside it, when the actual measured numbers show FlyHash costing MORE
under this project's own comparison code, would be exactly the kind of
overclaiming this project has spent 11 stages catching and correcting in
itself. This goes in the paper as a named, scoped limitation and an
explicit direction for future work, not a footnote.

---

## Stage 13: trying to close the speed/memory gap — a genuine partial win

**Issue.** Stage 12 found FlyHash's dense-packed comparison to be slower
and larger than SimHash's at equal `k`, and flagged a sparse-aware
comparison as the obvious next thing to try. Before accepting that as a
closed limitation, it was worth actually attempting the fix rather than
just naming it as future work — a real chance the gap would close, at
least partially.

**Investigation.** The key fact that makes a sparse comparison possible:
every FlyHash code has *exactly* k active bits (that's the definition of
k-winners-take-all), so for two such codes, `hamming(a,b) = 2k -
2·intersection(a,b)`. Since k is fixed, ranking by Hamming distance is
mathematically identical to ranking by descending intersection size — so
the comparison can be done as a sparse-sparse matrix product
(`scipy.sparse`) over just the k active indices per code, instead of a
dense matmul over the full pool_size=2000-wide array. Built
`sparse_flyhash_benchmark.py` to test this directly against the dense
version and against SimHash, at the same 5 bit budgets on GloVe.

**Resolution.** A real, non-obvious, mixed result:
- **Speed**: dramatic improvement over the dense comparison (10-22x
  faster: e.g. at k=16, 15.77ms dense -> 0.71ms sparse), and now correctly
  *scales with k* rather than being a flat ~14ms regardless of bit budget.
  But SimHash's own dense k x k matmul is still 2-5x faster than the
  sparse-intersection approach at every k tested — sparse-matrix format
  overhead in scipy dominates at these small scales (k up to 256), even
  though the underlying operation should in principle be cheaper.
- **Memory**: also now scales correctly with k (32 to 512 bytes) instead
  of a flat 250 bytes, a real behavioral improvement — but storing each
  active index costs a full 2-byte integer, so it's still ~16x larger
  than SimHash's bit-packed k-bit code. Bit-packing is simply a more
  efficient encoding than storing explicit positions, when the pool is
  this size (~2000): 1 bit per active position vs. 16 bits per index.

**Conclusion.** This genuinely moves the needle without fully closing it.
The honest updated claim: a sparse-aware comparison recovers most (not
all) of FlyHash's speed disadvantage and improves memory's *scaling
behavior* without matching SimHash's *absolute* memory efficiency. This
is exactly the kind of finding worth having attempted and reported
precisely, rather than either (a) not trying and leaving a vaguer "future
work" note, or (b) overselling a partial result as "solved." A smarter
index encoding (delta-coding the sorted indices, variable-length codes,
or reducing the pool size to shrink the index range) might close the
memory gap further — flagged as the next concrete thing to try if this
line of investigation continues, not attempted here to avoid an
open-ended rabbit hole right before the paper write-up.

---

## Stage 14: two figures for the paper draft — no new numbers, just visualization

**Issue.** The Phase 3 outline flagged two figures as "worth building" to
strengthen Sections 4.1 and 4.2 of the draft: a summary plot of FlyHash's
relative advantage across all four datasets at once, and heatmaps of the
Phase 2c hyperparameter grid (three text tables read slower than one
picture). Neither had been built while the draft's prose sections were
being written.

**Investigation/resolution.** Built `paper/figures/make_figures.py`, which
transcribes the existing numbers directly from README.md (no new runs, no
new numbers) and renders:

- **Figure 1** — line plot of the FlyHash/SimHash mAP ratio vs. bit budget,
  one line per dataset (GloVe, MiniLM, digits, MNIST), with a parity
  reference line at y=1. Confirms visually that the "biggest win at the
  smallest bit budget" pattern (Stage 8-10) holds across all four datasets
  simultaneously, and that MiniLM's curve is flatter (smaller relative
  advantage throughout) than the other three — a detail easy to miss
  reading four separate tables but immediately visible in one plot.
- **Figure 2** — three heatmaps (sparsity x pool_size, colored by mAP) for
  k=16/64/256 on GloVe, condensing Table 3's three text tables from Stage
  11. Makes the two secondary hyperparameter patterns (sparsity helps
  slightly at high k; pool size matters less past ~8x the bit budget)
  visible at a glance rather than requiring a cell-by-cell read.

One layout bug: the first render of Figure 2 had the shared suptitle
overlapping the middle panel's own title. Fixed by pushing the panels down
(`subplots_adjust(top=0.78)`) and moving the suptitle up (`y=1.05`) rather
than shrinking the panel titles or dropping the shared title.

**Conclusion.** Both figures are referenced inline in `paper/DRAFT.md`
(Sections 4.1 and 4.2) alongside their existing tables — the tables remain
the source of truth for exact numbers; the figures are for pattern
legibility, not a replacement. Saved as both PNG (for quick viewing) and
PDF (for eventual LaTeX/arXiv inclusion in Phase 4).

---

## Lessons learned (so far)

1. **A hypothesis that fits the data isn't the same as a hypothesis that's
   been checked against the data's provenance.** The non-negativity theory
   fit every result available at the time. It was still wrong, because the
   results themselves were downstream of an unexamined bug in the
   comparison method that produced them.
2. **When a result contradicts a published paper on the same dataset, the
   default assumption should be "my comparison is different from theirs,"
   not "the effect is real and paper-specific."** The eventual fix came
   from reading a reference implementation's actual parameterization, not
   from another round of hypothesis-and-test on this codebase alone.
3. **Ruling out an explanation is progress, even when it doesn't resolve
   the question.** Three ruled-out hypotheses (normalization, binary vs.
   signed weights, evaluation metric) narrowed the search space and made
   the eventual reference-implementation diff a targeted comparison
   instead of a blind one.
4. **Degenerate data (near-duplicate records, tiny corpora) produces
   numbers that look like a measurement but are closer to noise.** The
   25-sentence repeated-corpus issue in Stage 4 didn't cause a crash or an
   obviously wrong number — it caused a plausible-looking number that
   didn't mean what it appeared to mean.
5. **Environment and tooling friction (paths, shells, credentials) is real
   project time and worth recording**, if only so the same mistake isn't
   repeated by future-me on the next machine.
6. **"It ran once" and "it reproduces from a clean checkout" are different
   claims.** The multi-seed harness worked the first time it was written,
   then failed on a fresh environment because dependencies and the rest of
   the project tree weren't there. A result only counts once it's been run
   from a state that doesn't assume anything left over from a previous
   session — this is exactly the kind of gap a reviewer or a future
   collaborator would hit immediately.
7. **A win on one axis (quality) says nothing about another axis (memory,
   speed) — measure each one separately, honestly, and don't let one
   borrow credibility from the other.** It would have been easy to write
   "FlyHash wins" once and let that imply it wins on every dimension.
   Measuring memory/latency directly, after quality was already
   well-established, surfaced a real cost that the quality numbers alone
   would never have revealed.
