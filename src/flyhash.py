"""
flyhash.py — Locality-sensitive hashing inspired by the Drosophila olfactory circuit.

Biological analogy
-------------------
~50 odorant receptor neurons (ORNs)  ->  dense input vector  (e.g. a text/data embedding)
random, sparse projection             ->  ~2000 Kenyon cells (each wired to ~6 random inputs)
winner-take-all (top ~5% fire)         ->  sparse binary "tag" for the input

Similar inputs -> similar sparse random projections -> similar tags.
This is the mechanism described in Dasgupta, Stevens & Navlakha (2017),
"A neural algorithm for a fundamental computing problem" (Science).

This module reimplements it as a general-purpose sparse binary hash you can
drop in front of (or in place of) a dense-vector ANN index.
"""

from __future__ import annotations
import numpy as np


class FlyHash:
    """
    Random sparse binary projection + k-winners-take-all hashing.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (dense) embeddings.
    hash_dim : int
        Number of "Kenyon cells" — output hash dimensionality. Typically
        set MUCH larger than input_dim (the fly expands ~50 -> ~2000, a ~40x
        expansion). Expansion is what makes the sparse code more separable.
    sparsity : float
        Fraction of connections each Kenyon cell receives from the input
        layer (the fly uses ~6/50 ≈ 0.12). Controls how "local" each random
        projection is.
    wta_percent : float
        Fraction of hash_dim allowed to "fire" (be set to 1) per input.
        The fly keeps the top ~5%.
    seed : int
        Random seed, so the projection matrix is reproducible. In a fixed
        deployment this matrix is generated once and then frozen — it acts
        like a fixed set of "synaptic weights."
    """

    def __init__(
        self,
        input_dim: int,
        hash_dim: int = 2000,
        sparsity: float = 0.10,
        wta_percent: float = 0.05,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hash_dim = hash_dim
        self.sparsity = sparsity
        self.wta_percent = wta_percent
        self.k = max(1, int(round(wta_percent * hash_dim)))

        rng = np.random.default_rng(seed)

        # Sparse random projection matrix: each of the hash_dim output units
        # is connected to only `sparsity` fraction of the input dims, with
        # random +1/-1 weights on those connections (mirrors random,
        # non-negative-weighted, sparse ORN -> Kenyon cell wiring in the fly;
        # we allow +/-1 for better separation on continuous embeddings).
        num_connections = max(1, int(round(sparsity * input_dim)))
        self.projection = np.zeros((hash_dim, input_dim), dtype=np.float32)
        for i in range(hash_dim):
            idx = rng.choice(input_dim, size=num_connections, replace=False)
            weights = rng.choice([-1.0, 1.0], size=num_connections)
            self.projection[i, idx] = weights

    def hash(self, x: np.ndarray) -> np.ndarray:
        """
        Hash a single vector or a batch of vectors.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,) or (n, input_dim)

        Returns
        -------
        np.ndarray of {0,1}, shape (hash_dim,) or (n, hash_dim)
        """
        single = x.ndim == 1
        if single:
            x = x[None, :]

        # Project: (n, input_dim) @ (input_dim, hash_dim) -> (n, hash_dim)
        projected = x @ self.projection.T

        # k-winners-take-all: keep top-k activations per row, zero the rest.
        codes = np.zeros_like(projected, dtype=np.uint8)
        k = self.k
        # argpartition for efficiency instead of full sort
        top_k_idx = np.argpartition(-projected, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(projected.shape[0])[:, None]
        codes[rows, top_k_idx] = 1

        return codes[0] if single else codes

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
        """Hamming distance between two binary codes."""
        return int(np.count_nonzero(a != b))

    @staticmethod
    def pack_bits(codes: np.ndarray) -> np.ndarray:
        """Pack a (n, hash_dim) binary array into bytes for compact storage."""
        return np.packbits(codes, axis=-1)

    @staticmethod
    def unpack_bits(packed: np.ndarray, hash_dim: int) -> np.ndarray:
        """Inverse of pack_bits."""
        return np.unpackbits(packed, axis=-1)[..., :hash_dim]


class SimHash:
    """
    Standard random-hyperplane LSH (Charikar, 2002) — the conventional
    baseline FlyHash is compared against in the literature. Each output
    bit is the sign of a projection onto a random dense hyperplane
    (contrast with FlyHash's sparse projection + k-WTA).

    Given the SAME number of output bits, FlyHash reportedly preserves
    more neighborhood structure than SimHash because of its sparse,
    high-dimensional expansion (Dasgupta, Stevens & Navlakha, 2017,
    "A neural algorithm for a fundamental computing problem", Science).
    This class exists so you can verify that claim yourself.
    """

    def __init__(self, input_dim: int, hash_dim: int = 64, seed: int = 42):
        self.input_dim = input_dim
        self.hash_dim = hash_dim
        rng = np.random.default_rng(seed)
        self.hyperplanes = rng.normal(size=(hash_dim, input_dim)).astype(np.float32)

    def hash(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 1
        if single:
            x = x[None, :]
        projected = x @ self.hyperplanes.T
        codes = (projected > 0).astype(np.uint8)
        return codes[0] if single else codes
