"""
digits_embeddings.py — Real image data loader for Phase 2b, using scikit-learn's
built-in `digits` dataset (1797 handwritten digit images, 8x8 = 64 pixels each,
10 classes 0-9, derived from the UCI ML hand-written digits dataset).

Why this instead of full MNIST: full MNIST (60,000 28x28 images) isn't
reachable from this sandbox — the same network restriction that blocked
huggingface.co for real_embeddings.py also blocks every MNIST hosting source
tried (openml.org, storage.googleapis.com, s3.amazonaws.com all returned
403/connection errors; only GitHub and PyPI are reachable here, and no
verified GitHub-hosted MNIST CSV mirror was available to point at instead of
guessing at URLs). `sklearn.datasets.load_digits()` ships inside scikit-learn
itself, needs no network call, and is a genuine (if much smaller and lower-
resolution) real handwritten-digit image dataset — a legitimate stand-in
for validating the pixel-vector regime, not a synthetic substitute.

**This is not full MNIST.** If you want the real 60k-image, 28x28 MNIST
result, run this project on your own machine with internet access and swap
in `sklearn.datasets.fetch_openml("mnist_784")` or `torchvision.datasets.MNIST`
— same `load_embeddings(...)` interface, drop-in replacement. Flagged as a
pending item in the README, exactly like the real_embeddings.py situation.

Interface matches every other loader in this repo:
    load_embeddings(n_docs, embed_dim=None, seed=0, **_ignored) -> (texts, embeddings)
"""

from __future__ import annotations
import numpy as np


def load_embeddings(n_docs: int = 1797, embed_dim=None, seed: int = 0, **_ignored):
    """
    Loads scikit-learn's built-in digits dataset (up to 1797 images, 64 pixel
    dims each). `texts` are just human-readable labels ("digit 7, sample 12")
    since there's no natural text form for pixel data — kept only so this
    loader matches the same (texts, embeddings) interface as the others.
    """
    from sklearn.datasets import load_digits

    data = load_digits()
    X = data.data.astype(np.float32)   # (1797, 64), pixel intensities 0-16
    y = data.target

    if n_docs < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_docs, replace=False)
        X = X[idx]
        y = y[idx]
    elif n_docs > len(X):
        print(f"[digits_embeddings] Warning: requested n_docs={n_docs} exceeds "
              f"the dataset size ({len(X)}). Using all {len(X)} available images.")

    # Normalize pixel intensities to [0, 1] (raw range is 0-16).
    X = X / 16.0

    texts = [f"digit {label}, sample {i}" for i, label in enumerate(y)]
    return texts, X


if __name__ == "__main__":
    texts, embeddings = load_embeddings(n_docs=1797)
    print(f"Loaded {len(texts)} digit images, dim={embeddings.shape[1]}")
    print(f"Example: {texts[0]}, pixel range [{embeddings[0].min():.2f}, {embeddings[0].max():.2f}]")
