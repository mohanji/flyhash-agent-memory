"""
mnist_embeddings.py — Full MNIST (60,000 28x28 handwritten digit images, 784
pixel dims) via scikit-learn's fetch_openml.

RUN THIS ON YOUR OWN MACHINE, not in a network-locked sandbox: fetch_openml
needs to hit openml.org (~120MB download, cached locally after the first run
under ~/scikit_learn_data/).

This is the real MNIST the original FlyHash paper and its follow-ups
(BioHash, DenseFly) benchmark against — a step up from digits_embeddings.py's
sklearn `digits` dataset (1797 images, 8x8), which was used as a stand-in
when this sandbox couldn't reach any MNIST hosting source.

Same interface as every other loader in this repo:
    load_embeddings(n_docs, embed_dim=None, seed=0, **_ignored) -> (texts, embeddings)
"""

from __future__ import annotations
import numpy as np


def load_embeddings(n_docs: int = 5000, embed_dim=None, seed: int = 0, **_ignored):
    from sklearn.datasets import fetch_openml

    print("Loading MNIST (mnist_784 via OpenML, cached after first run)...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                         as_frame=False, parser="auto")
    X = X.astype(np.float32) / 255.0  # normalize pixel intensities to [0, 1]

    rng = np.random.default_rng(seed)
    if n_docs < len(X):
        idx = rng.choice(len(X), size=n_docs, replace=False)
        X = X[idx]
        y = np.asarray(y)[idx]
    elif n_docs > len(X):
        print(f"WARNING: n_docs={n_docs} exceeds MNIST's {len(X)} images; "
              f"using all {len(X)}.")
        y = np.asarray(y)

    texts = [f"digit {label}, sample {i}" for i, label in enumerate(y)]
    print(f"Loaded {len(texts)} MNIST images, dim={X.shape[1]}")
    return texts, X


if __name__ == "__main__":
    texts, embeddings = load_embeddings(n_docs=5000)
    print(f"Embedded {len(texts)} images -> shape {embeddings.shape}")
