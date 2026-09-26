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
