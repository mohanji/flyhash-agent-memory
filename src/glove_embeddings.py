"""
glove_embeddings.py — Loads real GloVe word vectors (Stanford NLP, via
gensim's cached downloader) as the embedding source for recall_vs_bits.py.

Why GloVe specifically: the original FlyHash paper (Dasgupta, Stevens &
Navlakha, 2017) tested on GloVe among its benchmark datasets and reported
FlyHash outperforming SimHash there. GloVe vectors are dense and SIGNED
(same category as sentence-transformer embeddings), so if FlyHash wins
here but lost on MiniLM sentence embeddings, that tells us the discrepancy
is about something specific to sentence-transformer / contrastive
embeddings — not about signed-vs-unsigned data in general.

Same interface as benchmark.load_embeddings() and real_embeddings.load_embeddings(),
so it's a drop-in swap for recall_vs_bits.py.
"""

from __future__ import annotations
import numpy as np


def load_embeddings(n_docs: int = 5000, embed_dim: int = None, seed: int = 0, **_ignored):
    """
    Loads the first n_docs most-frequent words' GloVe vectors (gensim's
    index_to_key is frequency-ordered, so this gives a mix of common words
    rather than random noise words, similar in spirit to how the original
    paper would have used a large, common vocabulary).
    `embed_dim` is ignored; glove-wiki-gigaword-100 is fixed at 100-dim.
    """
    import gensim.downloader as api
    print("Loading GloVe vectors (glove-wiki-gigaword-100, cached after first run)...")
    model = api.load("glove-wiki-gigaword-100")

    words = model.index_to_key[:n_docs]
    embeddings = np.array([model[w] for w in words], dtype=np.float32)
    texts = words  # "text" here is just the word itself
    print(f"Loaded {len(texts)} GloVe word vectors, dim={embeddings.shape[1]}")
    return texts, embeddings


if __name__ == "__main__":
    texts, embeddings = load_embeddings(n_docs=20)
    print(f"Sample words: {texts[:10]}")
    print(f"Embeddings shape: {embeddings.shape}")
