#!/usr/bin/env bash
# Reproduces the paper's experiments. Usage: ./reproduce.sh [quick|full]
#   quick: ~1-2 min sanity pass (2 seeds, small corpora)
#   full : paper settings (10 seeds); GloVe tables need the GloVe download (see REPRODUCING.md)
# v2 additions (appended below, v1 steps unchanged): equal-storage control; in full mode a LoCoMo run, which needs
#   data_external/locomo10.json (REPRODUCING.md, section 4a)
set -euo pipefail
MODE="${1:-quick}"
cd "$(dirname "$0")"
python -m pytest -q tests
python src/export_dataset.py --out datasets/agent_memory_v1 >/dev/null
cd src
if [ "$MODE" = "quick" ]; then SEEDS=2; else SEEDS=10; fi
echo "== Labeled agent-memory benchmark (paper: realistic-benchmark table) =="
python agent_memory_realistic_benchmark.py --seeds "$SEEDS"
echo "== Offline agent-memory demo =="
python offline_demo.py
echo "== Equal-storage control on the labeled benchmark (paper: labeled-storage table) =="
mkdir -p ../results
if [ "$MODE" = "quick" ]; then
  python labeled_equal_storage.py --seeds 2 --bootstrap 200 --out ../results/labeled_equal_storage.csv
else
  python labeled_equal_storage.py --seeds 10 --bootstrap 2000 --out ../results/labeled_equal_storage.csv
fi
if [ "$MODE" = "full" ]; then
  echo "== Digits (offline) main comparison + paired statistics =="
  python multi_seed_benchmark.py --dataset digits --seeds 10
  python paired_statistics.py --dataset digits --seeds 10
  python hnsw_systems_benchmark.py --dataset digits
  python weight_sensitivity_benchmark.py --dataset digits --seeds 10
  if [ -f ../data_external/locomo10.json ]; then
    echo "== LoCoMo evidence retrieval + equal-storage control (needs sentence-transformers) =="
    python locomo_recall.py --data ../data_external/locomo10.json --embedder minilm \
      --bits 16 32 64 128 --extra-simhash-bits 256 512 1024 2000 --k 5 10 20 50 --boot-k 10 --seeds 10 \
      --out ../results/locomo_recall_v2.csv
  else
    echo "Skipping LoCoMo: download locomo10.json to data_external/ first (see REPRODUCING.md, section 4a)."
  fi
fi
