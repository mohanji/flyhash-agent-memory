"""Smoke tests for the v2 analysis scripts (equal-storage control, LoCoMo recall).

They use tiny synthetic inputs and need no downloads, no LoCoMo data and no sentence-transformers model.
They check that the scripts run, write the expected files, and that the privacy-preserving dump holds ids only.
"""
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")


@pytest.fixture
def work_dir():
    """A scratch folder from tempfile, so the tests do not depend on pytest's own temp-dir (which can be unreadable
    on some Windows setups)."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _run(script, *args, cwd):
    return subprocess.run([sys.executable, os.path.join(SRC, script), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=600)


def _fake_locomo(path, n_conv=2, n_turn=60, n_q=50):
    rnd = random.Random(0)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()
    data = []
    for c in range(n_conv):
        turns = [{"dia_id": f"D1:{i + 1}", "speaker": "A", "text": " ".join(rnd.choices(words, k=8))}
                 for i in range(n_turn)]
        qa = []
        for q in range(n_q):
            i = rnd.randrange(n_turn)
            qa.append({"question": turns[i]["text"], "category": 1 + q % 5, "evidence": [f"D1:{i + 1}"]})
        data.append({"sample_id": f"s{c}", "conversation": {"session_1": turns}, "qa": qa})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_locomo_recall_smoke_and_ids_only_dump(work_dir):
    data = work_dir / "fake_locomo.json"
    _fake_locomo(data)
    out = work_dir / "out.csv"
    dump = work_dir / "dump"
    r = _run("locomo_recall.py", "--data", str(data), "--embedder", "bow", "--bits", "16", "32", "--k", "5", "10",
             "--seeds", "2", "--bootstrap", "100", "--boot-k", "10", "--center", "yes",
             "--dump-contexts", str(dump), "--dump-k", "10", "--dump-ids-only", "--out", str(out), cwd=str(work_dir))
    assert r.returncode == 0, r.stderr
    for name in ("out.csv", "out_bootstrap.csv", "out_percat_bootstrap.csv"):
        assert (work_dir / name).exists(), name
    files = sorted(p.name for p in dump.glob("*.jsonl"))
    assert "dense.jsonl" in files and "flyhash-b16.jsonl" in files and "simhash-b16.jsonl" in files
    with open(dump / "flyhash-b16.jsonl", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert set(rec) == {"qid", "category", "evidence_dia_ids", "ranked_dia_ids"}   # no dialogue text redistributed
    assert len(rec["ranked_dia_ids"]) == 10


def test_labeled_equal_storage_smoke(work_dir):
    out = work_dir / "les.csv"
    r = _run("labeled_equal_storage.py", "--n_personas", "20", "--seeds", "1", "--fly-bits", "16", "--sim-bits", "16", "64",
             "--bootstrap", "50", "--out", str(out), cwd=str(work_dir))
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert {r_["method"] for r_ in rows} == {"flyhash", "simhash"}
    assert all(0.0 <= float(r_["map_mean"]) <= 1.0 for r_ in rows)
    assert float(rows[0]["dense_map"]) > 0.0
    assert (work_dir / "les_bootstrap.csv").exists()
