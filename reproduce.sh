#!/usr/bin/env bash
# Reproduces the paper's experiments. Usage: ./reproduce.sh [quick|full]
#   quick: ~1-2 min sanity pass (2 seeds, small corpora)
#   full : paper settings (10 seeds); GloVe tables need the GloVe download (see REPRODUCING.md)
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
if [ "$MODE" = "full" ]; then
  echo "== Digits (offline) main comparison + paired statistics =="
  python multi_seed_benchmark.py --dataset digits --seeds 10
  python paired_statistics.py --dataset digits --seeds 10
  python hnsw_systems_benchmark.py --dataset digits
  python weight_sensitivity_benchmark.py --dataset digits --seeds 10
fi
