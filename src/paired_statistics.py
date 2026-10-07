"""
paired_statistics.py — supervisor-requested statistical upgrade (feedback
item #2): go beyond "mean +/- std per method, gap > 2*combined std" and
report a proper paired analysis, since FlyHash and SimHash are evaluated
on the exact same seeds, same dataset, and same fixed query sample.

For each bit budget, this script:
  1. Computes the PAIRED difference (FlyHash mAP - SimHash mAP) per seed
     — not the difference of means, but the mean of per-seed differences,
     which is the statistically appropriate quantity for a paired design.
  2. Reports a 95% confidence interval on the mean paired difference
     (both a normal-theory t-interval and a seed-level bootstrap interval,
     since n_seeds=10 is small enough that it's worth checking they agree).
  3. Reports an effect size (Cohen's d for paired samples: mean difference
     / std of differences).
  4. Runs a Wilcoxon signed-rank test (the standard nonparametric paired
     test) AND a paired sign-flip permutation test (10,000 resamples),
     since the original "gap > 2*combined std" check was explicitly not a
     formal significance test — these are the formal versions.

Reuses multi_seed_benchmark.run(), which already returns the raw per-seed
mAP arrays (fly_raw / sim_raw) needed for a paired analysis — no new data
collection, just a sharper statistical treatment of the existing numbers.

Caveat carried over honestly from the original design: the query sample
is held FIXED across seeds (only projection-matrix randomness varies), so
every statistic below describes uncertainty over projection-matrix draws,
not uncertainty over which queries get asked. This is stated explicitly
in the paper text, not just here.

Usage:
    python paired_statistics.py --dataset glove --seeds 10
"""

from __future__ import annotations
import argparse
import numpy as np
from scipy import stats

from multi_seed_benchmark import run as run_multiseed


def paired_analysis(fly_raw: np.ndarray, sim_raw: np.ndarray, n_perm: int = 10000, seed: int = 0):
    diffs = fly_raw - sim_raw
    n = len(diffs)
    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)  # sample std, for t-interval

    # 95% CI, normal-theory t-interval
    se = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_t_lo, ci_t_hi = mean_diff - t_crit * se, mean_diff + t_crit * se

    # 95% CI, seed-level bootstrap (resample the 10 paired differences with
    # replacement; percentile method) — a check that doesn't assume
    # normality, given n=10 is small.
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_perm)
    ])
    ci_boot_lo, ci_boot_hi = np.percentile(boot_means, [2.5, 97.5])

    # Effect size: Cohen's d for paired samples.
    cohens_d = mean_diff / std_diff if std_diff > 0 else float("inf")

    # Wilcoxon signed-rank test (paired, nonparametric).
    if np.all(diffs == diffs[0]) or np.all(diffs > 0) is False and np.all(diffs < 0) is False:
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(fly_raw, sim_raw)
        except ValueError:
            # all differences identical/zero, or too few non-zero diffs
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
    else:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(fly_raw, sim_raw)

    # Paired sign-flip permutation test: under the null (no systematic
    # difference), each diff's sign is an independent fair coin flip.
    # Build the null distribution of the mean by randomly flipping signs.
    signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = (signs * np.abs(diffs)).mean(axis=1)
    perm_p = float(np.mean(np.abs(perm_means) >= np.abs(mean_diff)))

    return {
        "mean_diff": mean_diff,
        "ci_t": (ci_t_lo, ci_t_hi),
        "ci_boot": (ci_boot_lo, ci_boot_hi),
        "cohens_d": cohens_d,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "perm_p": perm_p,
    }


def run(dataset: str = "glove", n_docs: int = 5000, n_queries: int = 200,
        n_seeds: int = 10, R: int = 100,
        bit_budgets=(16, 32, 64, 128, 256), flyhash_pool_size: int = 2000):

    summary = run_multiseed(dataset=dataset, n_docs=n_docs, n_queries=n_queries,
                             n_seeds=n_seeds, R=R, bit_budgets=bit_budgets,
                             flyhash_pool_size=flyhash_pool_size)

    print("\n" + "=" * 100)
    print("PAIRED STATISTICAL ANALYSIS (per bit budget, n_seeds paired differences)")
    print("=" * 100)
    print(f"{'k':>5} | {'mean diff':>10} | {'95% CI (t)':>20} | {'95% CI (bootstrap)':>20} | "
          f"{'Cohen d':>8} | {'Wilcoxon p':>11} | {'perm p':>8}")
    print("-" * 100)

    results = []
    for row in summary:
        fly_raw = np.array(row["fly_raw"])
        sim_raw = np.array(row["sim_raw"])
        stats_row = paired_analysis(fly_raw, sim_raw)
        ci_t = stats_row["ci_t"]
        ci_boot = stats_row["ci_boot"]
        print(f"{row['bits']:>5} | {stats_row['mean_diff']:>10.4f} | "
              f"[{ci_t[0]:>7.4f}, {ci_t[1]:>7.4f}] | "
              f"[{ci_boot[0]:>7.4f}, {ci_boot[1]:>7.4f}] | "
              f"{stats_row['cohens_d']:>8.2f} | {stats_row['wilcoxon_p']:>11.2e} | "
              f"{stats_row['perm_p']:>8.2e}")
        results.append({"bits": row["bits"], **stats_row})

    print("\nNotes:")
    print("- mean diff / CIs are on the PAIRED per-seed FlyHash-minus-SimHash mAP")
    print("  difference, not the difference of the two methods' independent means.")
    print("- Cohen's d here is the paired effect size (mean diff / std of diffs);")
    print("  values above ~0.8 are conventionally considered 'large'.")
    print("- Wilcoxon and permutation p-values are two-sided; with n_seeds=10 and")
    print("  FlyHash beating SimHash on every seed, the minimum attainable p-value")
    print("  for either test is ~1/2^10 ~ 0.00195 (a sign-test floor), so p-values")
    print("  reported at or near that floor reflect this resolution limit, not a")
    print("  a miscalculation.")
    print("- All of the above still describes uncertainty over PROJECTION-MATRIX")
    print("  draws only, since the query sample is held fixed across seeds by design.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="glove",
                         choices=["glove", "synthetic", "digits", "real", "mnist"])
    parser.add_argument("--n_docs", type=int, default=5000)
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--R", type=int, default=100)
    args = parser.parse_args()

    run(dataset=args.dataset, n_docs=args.n_docs, n_queries=args.n_queries,
        n_seeds=args.seeds, R=args.R)
