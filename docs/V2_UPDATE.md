# Update (v2, November 2026): equal-storage controls and real conversational data

This is a dated, additive update. **Nothing from the original release was removed.** The original code, dataset,
results, README findings and documentation stay as published; the original release is preserved as the git tag `v1.0`,
the Zenodo archive of that release, and arXiv v1 of [2611.05791](https://arxiv.org/abs/2611.05791). v2 adds new
experiments, new result files, new tests and a refined statement of the claim. Where the wording of the claim is
refined, this file says so.

## Why an update

Version 1 compared FlyHash with SimHash at equal **active bits**. That is the protocol of the original FlyHash
literature, and every v1 result stands. It does not hold **storage** equal: a FlyHash code with k active units out of
m = 2,000 costs about k x 11 bits as an index list (or 2,000 bits as a bitmap), whereas a SimHash code costs its bit
count. v2 adds that control, and a test on real conversations (LoCoMo) in addition to our templated benchmark.

## What the update found

| Question | Answer |
|---|---|
| Does FlyHash beat SimHash at the same active bits on real conversations? | Yes. On LoCoMo (1,535 answerable questions, MiniLM), recall@10 is higher by +0.10 to +0.15 at 16 to 128 active bits, ahead in 10 of 10 conversations (paired bootstrap, 2,000 resamples). |
| At equal storage? | SimHash is as good or better. SimHash with about 4x the bits matches FlyHash (differences within 0.01 at recall@10), and SimHash with 1,024 bits is ahead of FlyHash at 128 active bits by 0.024 (95% interval -0.030 to -0.018). The labeled benchmark shows the same pattern. |
| How far behind is hashing compared with exact dense search? | Dense cosine on the unhashed embeddings stays ahead of every hashed code. SimHash with 1,024 bits keeps about 97% of dense recall at recall@50 (0.651 vs 0.671, raw embeddings). |
| Does FlyHash lose more on some question types? | No evidence of it. As a share of dense recall at 128 active bits (mean-centred): multi-hop 0.878, temporal 0.903, open-domain 0.887, single-hop 0.913; every difference from single-hop has an interval that includes zero. |

So the claim of the paper is stated more precisely: **FlyHash's advantage is per active bit, not per byte.** This
refines, and does not retract, the v1 statements; see "Wording" below.

## New and changed files

All additions; no v1 file was deleted.

| File | What it is |
|---|---|
| `src/locomo_recall.py` | LoCoMo evidence-retrieval experiment (FlyHash, SimHash, dense cosine), with paired bootstrap, an equal-storage control (`--extra-simhash-bits`), per-category intervals, and an ids-only context dump (`--dump-ids-only`). |
| `src/labeled_equal_storage.py` | Equal-storage control on the labeled agent-memory benchmark, with dense reference and paired bootstrap. |
| `scripts/make_v2_figures.py` | Draws the two new figures from the result CSVs. |
| `results/v2/*.csv` | Aggregate result files (numbers only; no LoCoMo text). |
| `docs/images/fig3_equal_storage.png`, `fig4_locomo_detail.png` | New figures. |
| `tests/test_v2_scripts.py` | Smoke tests for both scripts on synthetic data (no downloads). |
| `REPRODUCING.md` | New rows and section 4a (additions only). |
| `reproduce.sh` | New steps appended (the v1 steps are unchanged). |

## Reproducing the update

```bash
# equal-storage control on the labeled benchmark (about a minute with --seeds 2, longer with 10)
python src/labeled_equal_storage.py --seeds 10 --bootstrap 2000 --out results/v2/labeled_equal_storage.csv

# LoCoMo (download locomo10.json into data_external/ first; see REPRODUCING.md, section 4a)
python src/locomo_recall.py --data data_external/locomo10.json --embedder minilm --bits 16 32 64 128 \
  --extra-simhash-bits 256 512 1024 2000 --k 5 10 20 50 --boot-k 10 --seeds 10 --out results/v2/locomo_recall_v2.csv

# figures
python scripts/make_v2_figures.py --locomo results/v2/locomo_recall_v2.csv \
  --locomo-boot results/v2/locomo_recall_v2_bootstrap.csv --labeled results/v2/labeled_equal_storage.csv --out docs/images

# tests (synthetic data, no downloads)
python -m pytest -q tests
```

The matched-bit rows of the labeled control reproduce the v1 benchmark table to within about 0.002 mAP, which checks
that the new script and the v1 script measure the same thing.

## Data and licence

LoCoMo (Maharana et al., 2024) is released under CC BY-NC. This repository does **not** redistribute it: download it
yourself (REPRODUCING.md, section 4a). The result files are aggregate numbers. `--dump-contexts` writes retrieved
dialogue text and must stay out of git (`contexts/` is ignored); `--dump-ids-only` writes turn ids only, so the output
can be shared without redistributing the text.

## Wording: where v2 refines v1

The v1 text is kept. The refinements are:

| Where | v1 wording | v2 clarification |
|---|---|---|
| Protocol | "matched-bit" / "same bit budget" | The budget that is matched is **active** bits. Storage is not matched; see above. |
| Claim | "a quality-per-bit result" | A quality-per-**active-bit** result; at equal storage SimHash is as good or better. |
| Paper | RQ3 and abstract | Restated per active bit; the paper (arXiv v2) lists every change in its own change log. |

## Limits of the update

- Both new benchmarks measure **retrieval**, not end-to-end answer accuracy.
- The labeled benchmark is templated and synthetic (150 queries); its intervals understate real-world variability.
- LoCoMo conversations are small (370 to 690 turns), where an exhaustive scan is cheap. An advantage of sparse codes at
  scale (for example with an inverted index over active units) is **not** tested here.
- Bootstrap intervals are over questions, which are correlated within a conversation; the per-conversation counts are
  reported as a robustness check.
- LongMemEval has not been run.
