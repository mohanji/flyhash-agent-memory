"""
map_metric.py — Mean Average Precision @ R (mAP@R), the standard evaluation
metric used in the FlyHash literature (Dasgupta et al. 2017; the BioHash
paper, Ryali et al. 2020, explicitly confirms this: "Following previous
work (Dasgupta et al., 2017; ...), we use Mean Average Precision (mAP) as
the evaluation metric.").

This differs from plain recall@k overlap: mAP rewards a method for ranking
TRUE near-neighbors EARLY in its retrieved list, not just for including
them somewhere in the top-k. A method that gets 5 true neighbors into
ranks 1-5 scores higher than one that gets the same 5 scattered across
ranks 1-50, even though both would show identical recall@50.

Definition (matching Eq. 12-13 in the BioHash paper):
    AP(q)@R = (1 / num_relevant) * sum_{l=1}^{R} Precision(l) * Rel(l)
    mAP@R = mean of AP(q)@R over all queries q

where Rel(l) = 1 if the item at rank l is a true near-neighbor (relevant),
and Precision(l) = (# relevant found in top l) / l.
"""

from __future__ import annotations
import numpy as np


def compute_map_at_r(pred_dists: np.ndarray, exact_dists: np.ndarray, R: int) -> float:
    """
    pred_dists, exact_dists: (n_queries, n_corpus) distance matrices.
    "Relevant" for a query = the R nearest neighbors under exact_dists.
    Returns mAP@R averaged over all queries.
    """
    n_queries = pred_dists.shape[0]
    relevant_sets = np.argsort(exact_dists, axis=1)[:, :R]  # (n_queries, R)
    pred_ranking = np.argsort(pred_dists, axis=1)[:, :R]     # (n_queries, R)

    aps = np.zeros(n_queries)
    for q in range(n_queries):
        relevant = set(relevant_sets[q].tolist())
        num_relevant = len(relevant)
        if num_relevant == 0:
            continue
        hits = 0
        precision_sum = 0.0
        for rank, idx in enumerate(pred_ranking[q], start=1):
            if idx in relevant:
                hits += 1
                precision_sum += hits / rank
        aps[q] = precision_sum / num_relevant

    return float(np.mean(aps))
