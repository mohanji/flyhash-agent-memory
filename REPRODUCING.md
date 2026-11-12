# Reproducing the paper

Paper: *FlyHash for LLM Agent Memory: An Empirical Validation and a New Application*
(Bheemadolu & Odhiambo, 2026).

## 1. Setup (about 2 minutes)

```bash
git clone https://github.com/mohanji/flyhash-agent-memory.git
cd flyhash-agent-memory
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                              # 4 tests, checks dataset + FlyHash
```

Tested with Python 3.10, numpy 2.2, scipy 1.15, scikit-learn 1.7, faiss-cpu 1.15.

## 2. The released dataset

`datasets/agent_memory_v1/` contains the labeled agent-memory benchmark from the paper:

| File | Contents |
|---|---|
| `memories.jsonl` | 6,000 memory sentences: `{"id", "text"}` |
| `queries.jsonl` | 150 questions with ground truth: `{"id", "text", "relevant_memory_ids"}` |
| `meta.json` | generation parameters (150 personas, seed 0) and summary statistics |

It is synthetic (templates + seeded RNG, no personal data) and fully deterministic.
Regenerate it, or build variants, with `python src/export_dataset.py --help`.
`tests/test_reproducibility.py` checks the released files match the generator.

## 3. One-command run

```bash
./reproduce.sh quick    # ~1 minute: tests + labeled benchmark (2 seeds) + offline demo
./reproduce.sh full     # paper settings (10 seeds) plus the offline Digits experiments
```

## 4. Paper table to command (run from `src/`)

| Paper table / section | Command | Needs |
|---|---|---|
| Labeled agent-memory benchmark (`tab:realistic-benchmark`) | `python agent_memory_realistic_benchmark.py --seeds 10` | core only |
| Main results (`tab:main-results`) | `python multi_seed_benchmark.py --dataset {glove,digits,mnist,real,synthetic} --seeds 10` | GloVe: `gensim`; MNIST: internet (OpenML) |
| Paired statistics (`tab:paired-stats`) | `python paired_statistics.py --dataset glove --seeds 10` | as above |
| Hyperparameter grid (`tab:hyperparam`) | `python grid_search.py` | core only |
| Memory and latency (`tab:memlatency`) | `python latency_memory_benchmark.py` | core only |
| Sparse-set ranking (`tab:sparse`) | `python sparse_flyhash_benchmark.py` | core only |
| End-to-end latency (`tab:e2e-latency`) | `python end_to_end_latency.py` | `sentence-transformers` |
| HNSW systems comparison (`tab:hnsw`) | `python hnsw_systems_benchmark.py --dataset glove` | core only (+ GloVe) |
| Weight-mode sensitivity (`tab:weight-mode`, Appendix A) | `python weight_sensitivity_benchmark.py --dataset glove --seeds 10` | as above |
| Equal-storage control, labeled benchmark (`tab:labeled-storage`) | `python labeled_equal_storage.py --seeds 10 --bootstrap 2000` | core only |
| LoCoMo evidence retrieval and equal-storage control (LoCoMo section) | `python locomo_recall.py --data ../data_external/locomo10.json --embedder minilm --bits 16 32 64 128 --extra-simhash-bits 256 512 1024 2000 --k 5 10 20 50 --boot-k 10 --seeds 10 --out locomo_recall_v2.csv` | LoCoMo download (section 4a), `sentence-transformers` |

Exact numbers depend on library versions and hardware; latency and memory figures in
particular vary by machine. Retrieval-quality numbers (mAP) are seeded and should match
closely.

### 4a. LoCoMo data (not included in this repository)

LoCoMo (Maharana et al., 2024) is released under CC BY-NC, so this repository does not redistribute it.
Download `locomo10.json` into the git-ignored `data_external/` folder:

```bash
mkdir -p data_external
curl -L -o data_external/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
# Windows PowerShell: Invoke-WebRequest -Uri "<same URL>" -OutFile "data_external\locomo10.json"
```

Expected load message: `10 conversations, 5882 turns, 1981 questions with usable evidence` (5 questions are skipped
for missing or unmatched evidence). The paper's "all" column excludes the adversarial category (1,535 questions);
change this with `--exclude-from-all`.

`locomo_recall.py --embedder bow` runs offline with a hashed bag-of-words embedding. It is a smoke test only and its
numbers are not meaningful. `--dump-contexts DIR` writes retrieved LoCoMo text, so keep that folder out of git
(`.gitignore` already lists `contexts/` and `data_external/`).

The result files are aggregate numbers only (`locomo_recall_v2.csv`, `locomo_recall_v2_bootstrap.csv`,
`labeled_equal_storage.csv`, `labeled_equal_storage_bootstrap.csv`). The matched-bit rows of
`labeled_equal_storage.py` reproduce `tab:realistic-benchmark` to within about 0.002 mAP.

## 5. The agent-memory demo

- **Offline, no API key, no downloads:** `python src/offline_demo.py` loads the released
  dataset into a FlyHash-backed store (TF-IDF+SVD embeddings) and prints recalled memories
  next to the ground-truth labels.
- **MiniLM-backed wrapper:** `src/agent_memory.py` (`AgentMemory`) embeds with
  `all-MiniLM-L6-v2` (downloads the model once). `pip install -r requirements-optional.txt`.
- **Interactive Claude chat agent:** `src/agent_demo.py` uses `AgentMemory` as a tool.
  Needs `pip install anthropic` and `ANTHROPIC_API_KEY` set in your environment.

## 6. Cite

See `CITATION.cff`.
