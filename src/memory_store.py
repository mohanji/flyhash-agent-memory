"""
memory_store.py — Drop-in agent memory stores.

Both stores share the same interface:
    store.add(text, embedding)
    store.query(embedding, k) -> list of (text, score) sorted best-first

so an agent's "memory" tool can swap between them with no other code changes.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np

from flyhash import FlyHash

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


@dataclass
class MemoryRecord:
    text: str
    embedding: np.ndarray


class FlyMemoryStore:
    """
    Sparse-binary memory store using FlyHash.

    Two retrieval modes:

    - mode="scan": exact Hamming distance over every stored code (correct,
      but O(n) per query — same complexity class as a flat index, so at
      small n a hand-rolled numpy scan won't beat FAISS's optimized C++).

    - mode="bucket": inverted-index lookup. Each of the k "active" Kenyon
      cells for a record is an index into a bucket list; a query only
      compares against records sharing at least one active cell, then
      ranks that (much smaller) candidate set by Hamming distance. This
      is the actual computational payoff of a sparse code — sublinear
      candidate retrieval instead of scanning the whole store — and is
      the mode that matters as the memory store grows large.

    Storage: only the k active bit-indices are kept per record (sparse
    representation), not a dense hash_dim-bit vector — this is what
    realizes the memory savings from a sparse, high-dimensional code.
    """

    def __init__(self, input_dim: int, hash_dim: int = 2000,
                 sparsity: float = 0.10, wta_percent: float = 0.05, seed: int = 42,
                 mode: str = "bucket"):
        self.hasher = FlyHash(input_dim, hash_dim, sparsity, wta_percent, seed)
        self.hash_dim = hash_dim
        self.mode = mode
        self.texts: list[str] = []
        self._active_indices: list[np.ndarray] = []  # sparse storage: k indices/record
        self._buckets: dict[int, list[int]] = {}     # cell_id -> [record_ids]

    def _hash_to_active_indices(self, code: np.ndarray) -> np.ndarray:
        return np.nonzero(code)[0].astype(np.uint16)

    def add(self, text: str, embedding: np.ndarray) -> None:
        code = self.hasher.hash(np.asarray(embedding, dtype=np.float32))
        active = self._hash_to_active_indices(code)
        record_id = len(self.texts)
        self.texts.append(text)
        self._active_indices.append(active)
        for cell in active:
            self._buckets.setdefault(int(cell), []).append(record_id)

    def add_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        codes = self.hasher.hash(np.asarray(embeddings, dtype=np.float32))
        for text, code in zip(texts, codes):
            active = self._hash_to_active_indices(code)
            record_id = len(self.texts)
            self.texts.append(text)
            self._active_indices.append(active)
            for cell in active:
                self._buckets.setdefault(int(cell), []).append(record_id)

    def _hamming_to_all(self, query_active: set[int]) -> np.ndarray:
        k = self.hasher.k
        dists = np.full(len(self.texts), 2 * k, dtype=np.int32)
        for i, active in enumerate(self._active_indices):
            overlap = len(query_active.intersection(active.tolist()))
            # symmetric-difference size = Hamming distance for equal-k sparse codes
            dists[i] = 2 * k - 2 * overlap
        return dists

    def query(self, embedding: np.ndarray, k: int = 5):
        if not self.texts:
            return []
        query_code = self.hasher.hash(np.asarray(embedding, dtype=np.float32))
        query_active = set(self._hash_to_active_indices(query_code).tolist())

        if self.mode == "scan":
            dists = self._hamming_to_all(query_active)
            top_k_idx = np.argsort(dists)[:k]
            return [(self.texts[i], int(dists[i])) for i in top_k_idx]

        # mode == "bucket": gather candidates sharing >=1 active cell only.
        candidate_ids: dict[int, int] = {}  # record_id -> overlap count
        for cell in query_active:
            for rid in self._buckets.get(cell, ()):
                candidate_ids[rid] = candidate_ids.get(rid, 0) + 1

        if not candidate_ids:
            return []

        code_k = self.hasher.k
        scored = [
            (rid, 2 * code_k - 2 * overlap)
            for rid, overlap in candidate_ids.items()
        ]
        scored.sort(key=lambda pair: pair[1])
        top = scored[:k]
        return [(self.texts[rid], dist) for rid, dist in top]

    def memory_bytes(self) -> int:
        # 2 bytes per active-cell index (uint16), k indices per record.
        return sum(a.nbytes for a in self._active_indices) + \
               sum(len(v) * 4 for v in self._buckets.values())  # bucket overhead (int refs)


class FaissMemoryStore:
    """Baseline: exact-search dense-vector store (ground truth for recall)."""

    def __init__(self, input_dim: int):
        if not _HAS_FAISS:
            raise ImportError("faiss-cpu not installed")
        self.index = faiss.IndexFlatL2(input_dim)
        self.texts: list[str] = []
        self._dim = input_dim

    def add(self, text: str, embedding: np.ndarray) -> None:
        self.index.add(np.asarray(embedding, dtype=np.float32)[None, :])
        self.texts.append(text)

    def add_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        self.index.add(np.asarray(embeddings, dtype=np.float32))
        self.texts.extend(texts)

    def query(self, embedding: np.ndarray, k: int = 5):
        q = np.asarray(embedding, dtype=np.float32)[None, :]
        dists, idx = self.index.search(q, k)
        return [(self.texts[i], float(dists[0][j])) for j, i in enumerate(idx[0]) if i != -1]

    def memory_bytes(self) -> int:
        return len(self.texts) * self._dim * 4  # float32
