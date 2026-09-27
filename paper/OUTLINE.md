# Paper outline (Phase 3 draft — for review before expanding to full prose)

Working title (open to changing):
**"FlyHash for LLM Agent Memory: An Empirical Validation, a Cautionary Tale,
and a New Application"**

(Alt: "Bio-Inspired Sparse Hashing for LLM Agent Memory Retrieval: A
Reproducibility Study with a New Application")

Target: arXiv preprint first (Phase 4), reusable as-is or shortened for
ACM XRDS (Phase 5). Written for that audience from the start so Phase 4
is close to copy-paste, not a rewrite.

**Author:** Mohan Bheemadolu, Independent Researcher.

---

## Abstract (~150-200 words)

- One sentence: what FlyHash is and why agent memory is a good fit for it.
- One sentence: what's actually new here (the application + the validation
  rigor — NOT a new algorithm; BioHash/DenseFly already improve on the base
  algorithm itself).
- One sentence: the Phase 1 discrepancy — found, diagnosed, fixed, and why
  that matters for anyone else doing this comparison.
- One sentence: what was validated (4 datasets, 2 modalities, 10 seeds,
  36 hyperparameter configs) and the headline result (robust win, largest
  at the smallest bit budgets).
- One sentence: the honest limitation — a naive brute-force comparison
  shows FlyHash's quality advantage does not translate into a memory or
  speed advantage over SimHash; a sparse-set comparison we also tried
  partially closes the speed gap but not the memory gap, so this remains
  an open, precisely-scoped trade-off rather than a solved or hidden issue.

## 1. Introduction

- The problem: LLM agents need persistent memory; dense-vector retrieval
  (FAISS, embeddings) is the default, but is memory/latency-heavy at scale.
- The biological hook: fly olfactory circuit as a compact, sparse hashing
  scheme (Dasgupta et al. 2017) — brief, not overclaimed.
- Why apply it to agent memory specifically: no prior work found doing
  this (cite the literature scan); a natural fit because agent memories
  are exactly the "many short text snippets, retrieved by similarity"
  setting FlyHash targets.
- Contributions, stated plainly:
  1. First application (to our knowledge) of FlyHash-style hashing to
     LLM agent memory retrieval, with a working demo.
  2. A precise statement of the fair-comparison protocol (hold the number
     of *winning* bits `k` equal, not total hash dimensionality `m`) —
     motivated by a real discrepancy we hit reproducing the original
     paper's GloVe result, traced to exactly this conflation (full
     investigation in Appendix A).
  3. Statistical validation the original paper doesn't show at this
     depth: multi-seed variance, 4 datasets across 2 modalities, and a
     36-configuration hyperparameter ablation.
  4. Honest limitations: no sublinear indexing demonstrated yet.

## 2. Related Work

- **Dasgupta, Sharma & Navlakha (2017)**, Science — the original algorithm;
  describe the 3-step mechanism (sparse random projection, k-WTA, sparse
  binary output) and their reported results (SIFT, GloVe, MNIST).
- **Sharma & Navlakha (2018)**, "DenseFly" (arXiv:1812.01844) — densifies
  the fly's sparse output for further gains; note this is a genuine
  algorithmic improvement this project does NOT implement (future work).
- **Ryali et al. (2020)**, "BioHash" (ICML) — replaces the fly's *random*
  projection with a *learned* one; established mAP as the standard metric
  (cite explicitly, since this project's metric choice follows them).
  Also a stronger technical contribution than this project — worth being
  upfront that BioHash > FlyHash empirically in their own results.
- **AAAI 2022, "FlyNN"** (arXiv:2112.07157) — applies the fly circuit to
  federated nearest-neighbor classification, not hashing-for-retrieval;
  shows the breadth of problems this circuit has been adapted to.
- **`FlyHash` PyPI package** (Huang, MIT license) — a general-purpose
  implementation exists; this project's contribution is the application
  and validation, not a new implementation to replace it.
- One paragraph tying it together: the gap this paper fills is the
  intersection of (agent memory) x (rigorous multi-dataset, multi-seed
  validation), which none of the above cover.

## 3. Method

### 3.1 FlyHash and SimHash (brief, formal)
- Equations/pseudocode for both (already have working code — translate to
  paper notation): sparse projection + k-WTA vs. dense random hyperplane.
- Define `k` (winning bits) vs. `m` (Kenyon-cell pool size) explicitly —
  this distinction is the crux of Section 4's discrepancy story, so it
  needs to be crisp here.

### 3.2 Evaluation protocol
- mAP@R definition (cite BioHash for using this metric).
- The **fair-comparison protocol**, stated crisply as the citable claim of
  this paper: hold `k` (winning bits) equal between FlyHash and SimHash,
  sweep `k`, keep `m` (pool size) fixed and large. One sentence flagging
  that naive comparisons which instead fix total hash dimensionality
  while holding `wta_percent` fixed conflate `k` and `m`, silently
  disadvantaging FlyHash — pointing to Appendix A for the full story of
  how this was discovered.

### 3.3 Datasets
- Table: dataset, modality, size, dimensionality (GloVe, MiniLM, sklearn
  digits, MNIST). One paragraph each on why chosen / what's a stand-in
  for what (digits as MNIST substitute where relevant, honestly labeled).

### 3.4 Statistical protocol
- Multi-seed procedure: what varies (projection matrices) vs. what's
  fixed (dataset, query sample) across seeds, and why that isolates the
  right source of variance.
- The "gap > 2×combined std" robustness criterion — simple, state its
  limits (not a formal significance test, just a conservative sanity
  check) rather than overselling it as rigorous statistics.

### 3.5 Agent memory integration
- Architecture: text -> sentence-transformer embedding -> FlyHash ->
  sparse code -> store; query -> same pipeline -> Hamming ranking.
- The Claude-API tool-use demo (`save_memory`/`recall_memory`), one
  concrete transcript example (the peanuts/Python example) as a figure.

## 4. Results

### 4.1 Main result: corrected comparison across datasets
- Table per dataset (GloVe, MiniLM, digits, MNIST) — mean ± std over 10
  seeds, all 5 bit budgets, robustness flag. (Have all 4 tables already,
  from README.)
- One summary figure: relative advantage (FlyHash/SimHash ratio) vs. bit
  budget, overlaid for all 4 datasets — shows the "biggest win at
  smallest budget" pattern visually, across datasets at once. (New —
  needs to be built; would be a strong figure.)

### 4.2 Hyperparameter ablation
- The 3-table (k=16/64/256) grid search results from README, or condense
  into 1-2 heatmap figures (sparsity x pool_size, colored by mAP, with
  SimHash baseline as a reference line/contour). (New figure, worth
  building — heatmaps read faster than 3 text tables in a paper.)

### 4.3 Memory and query latency — quality gains do not imply resource gains
- **Done, real numbers in hand** (`latency_memory_benchmark.py`, GloVe):
  table of packed bytes/record and ms/query for FlyHash vs. SimHash vs.
  exact L2, at all 5 bit budgets.
- State plainly: FlyHash's dense-packed code (pool_size=2000 wide, only k
  bits active) is LARGER (~250 bytes flat, regardless of k) and SLOWER
  (~13ms/query flat, 10-70x slower than SimHash at equal k) to
  brute-force-compare than SimHash's genuinely k-bit-wide code. This is
  the honest, structural finding — not spun, not buried.
- Explain why this doesn't undercut Section 4.1's quality results: mAP
  depends only on which k bits are set, not on storage/comparison format.

### 4.4 Closing the gap: a sparse-set comparison (partial success)
- **Done, real numbers in hand** (`sparse_flyhash_benchmark.py`): every
  FlyHash code has exactly k active bits, so `hamming(a,b) = 2k -
  2·intersection(a,b)` — ranking by Hamming distance is mathematically
  identical to ranking by descending intersection size, computable as a
  sparse-sparse matrix product over just the k active indices.
- Result table (dense vs. sparse vs. SimHash ms/query; sparse-index vs.
  SimHash bytes/record) at all 5 bit budgets.
- Honest result: recovers most (10-22x) of the speed gap and fixes
  memory's *scaling behavior* (now scales with k, not flat), but SimHash's
  small dense k×k matmul is still 2-5x faster, and index-list storage
  (2 bytes/index) is still ~16x larger than SimHash's bit-packed code.
  **A genuine partial win, explicitly not oversold as solving the
  problem.**

### 4.5 Agent memory demo
- Qualitative: the cross-turn recall transcript (the peanuts/Python
  example) as a figure — this is the "does it actually work for the
  target use case" evidence, separate from the benchmark numbers above.

## 5. Discussion

- Why is the quality advantage biggest at low bit budgets? (sparse
  expansion preserves more structure per bit than dense projection at the
  same bit count — the intuition from the original paper).
- The quality/resource split from 4.3-4.4 deserves its own paragraph here:
  this is a case where a hash function can be a clear win on one axis
  (nearest-neighbor structure preservation) while a naive implementation
  leaves resource costs worse, and even a targeted fix (the sparse-set
  comparison) only partially closes that gap — worth discussing as a
  general point about evaluating hashing schemes, not just as a
  limitation of this project specifically.
- Practical implication for agent memory: an agent choosing FlyHash today
  gets meaningfully better retrieval quality per bit; a well-implemented
  SimHash baseline remains competitive or better on raw memory and speed
  at the record counts tested here, so the deployment trade-off is
  quality vs. resource cost, not a strict win either way (Section 6).

## 6. Limitations and Future Work

- **Memory/speed under brute-force comparison do not fully match the
  quality advantage, even after a genuine attempt to close the gap** — see
  4.3-4.4. The sparse-set comparison recovers most of the speed
  disadvantage and fixes memory's scaling behavior, but SimHash's
  bit-packed code remains more memory-efficient, and a truly sublinear
  index (banded multi-probe LSH, or an inverted index proven sublinear at
  scale — unlike this project's earlier "bucket mode" attempt) is still
  unbuilt. Concrete next step identified but not attempted: a more compact
  index encoding (delta-coding sorted indices, variable-length codes, or a
  smaller pool) to close the remaining memory gap.
- SIFT1M and further datasets — explicitly deferred, not attempted.
- Doesn't implement DenseFly's densification or BioHash's learned
  projection — both are natural extensions.
- Small-corpus limits on the real MiniLM benchmark (100 sentences) —
  should be scaled up for a stronger real-text result.

## 7. Conclusion

- Restate contributions plainly, no overclaiming.

## References

- Full bibliography with the 5 sources found above, in whatever citation
  style the target venue wants (arXiv has no fixed style; can use a
  simple numbered format).

## Appendix A: Reproducibility Investigation

*(Moved here from the main body — keeps the full transparency/rigor value
without diluting the crisp technical claim in Section 3.2. This is where
readers doing their own reproduction, or reviewers who want to verify the
rigor, will look.)*

- State the discrepancy plainly: initial results contradicted the
  original paper's reported GloVe win for FlyHash.
- Walk through the 3 ruled-out hypotheses (normalization; signed vs.
  binary weights; recall@k vs. mAP@R) — briefly, each with the one-line
  reason it was ruled out.
- The actual bug: diffing against github.com/dataplayer12/Fly-LSH,
  finding the `k` vs. `m` conflation, explained in full with before/after
  numbers.
- **Explicit retraction**: the "FlyHash needs non-negative/redundant
  data" hypothesis from earlier work was wrong, and why — a paragraph
  modeled on the retraction already written in RESEARCH_JOURNAL.md.
- Closing framing: documented in this much detail specifically because
  fair-comparison bugs in this exact space are apparently easy to make —
  the goal is helping the next person avoid the same mistake, not a
  confession for its own sake.

---

## Status: all open questions resolved — ready to expand into full prose

1. ~~Section 4 narrative vs. appendix~~ — **resolved**: full investigation
   moved to Appendix A; the crisp k-vs-m claim stays in Section 3.2 where
   it's citable.
2. ~~Fresh latency/memory benchmark~~ — **resolved**: run under the
   corrected protocol (`latency_memory_benchmark.py`). Result is an
   honest limitation, not a win — see Section 4.3/5/6 above. Documented
   in README and RESEARCH_JOURNAL.md (Stage 12).
3. ~~Author/affiliation~~ — **resolved**: Mohan Bheemadolu, Independent
   Researcher (see top of this file).

Next: expand this outline into full prose, section by section, starting
with Abstract + Introduction for review before continuing.
