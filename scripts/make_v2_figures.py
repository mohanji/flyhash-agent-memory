"""
make_v2_figures.py - figures for the v2 equal-storage analysis (LoCoMo and the labeled benchmark).

Inputs (aggregate result files written by locomo_recall.py and labeled_equal_storage.py):
    locomo_recall_v2.csv, locomo_recall_v2_bootstrap.csv
    labeled_equal_storage.csv, labeled_equal_storage_bootstrap.csv
Outputs:
    fig3_equal_storage.pdf/.png   recall (LoCoMo) and mAP (labeled) against code size, FlyHash vs SimHash vs dense
    fig4_locomo_detail.pdf/.png   LoCoMo per-category recall, and paired-bootstrap differences

Usage:  python make_v2_figures.py --locomo locomo_recall_v2.csv --locomo-boot locomo_recall_v2_bootstrap.csv \
            --labeled labeled_equal_storage.csv --out figures
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# palette: categorical slots 1 and 2 of the validated reference palette (blue, orange) + neutral inks
BLUE, ORANGE = "#2a78d6", "#eb6834"
ORANGE_LIGHT = "#f5b199"                 # tint of the SimHash hue for the lower-bit variant
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
IDX_BITS = math.ceil(math.log2(2000))   # 11 bits per active unit as an index list


def setup_fonts():
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Liberation Sans", "Arial", "Helvetica", "DejaVu Sans") if f in names), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family, "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.8,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
    })
    return family


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def bits_axis(ax, lo=12, hi=4600):
    ax.set_xscale("log", base=2)
    ax.set_xlim(lo, hi)
    ticks = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["16", "32", "64", "128", "256", "512", "1k", "2k", "4k"])
    ax.minorticks_off()


def curve_panel(ax, fly, sim, dense, ylabel, title, ann_k):
    """fly/sim: DataFrames with columns bits, mean, std. FlyHash is drawn twice: by active bits (dashed) and by
    index-list storage (solid)."""
    style(ax)
    bits_axis(ax)
    ax.axhline(dense, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(13, dense + 0.006, "dense cosine", ha="left", va="bottom", color=INK2, fontsize=8)
    # SimHash
    ax.fill_between(sim.bits, sim["mean"] - sim["std"], sim["mean"] + sim["std"], color=ORANGE, alpha=0.14, linewidth=0)
    ax.plot(sim.bits, sim["mean"], color=ORANGE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3)
    # FlyHash by storage (index list): x = 11 * k
    xs = fly.bits * IDX_BITS
    ax.fill_between(xs, fly["mean"] - fly["std"], fly["mean"] + fly["std"], color=BLUE, alpha=0.14, linewidth=0)
    ax.plot(xs, fly["mean"], color=BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=4)
    # FlyHash by active bits: x = k (what the matched-bit protocol compares)
    ax.plot(fly.bits, fly["mean"], color=BLUE, linewidth=1.4, linestyle=(0, (1.5, 2)), marker="o", markersize=5,
            markerfacecolor=SURFACE, markeredgecolor=BLUE, markeredgewidth=1.4, zorder=3)
    ax.set_xlabel("code size (bits)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=8)


def fig_equal_storage(lo, lab, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)

    # (a) LoCoMo: recall@10, mean-centred, all* (categories 1-4)
    d = lo[(lo.category.astype(str) == "all") & (lo.k == 10) & (lo.centered == True)]  # noqa: E712
    fly = d[d.method == "flyhash"].rename(columns={"recall_mean": "mean", "recall_std": "std"}).sort_values("bits")
    sim = d[d.method == "simhash"].rename(columns={"recall_mean": "mean", "recall_std": "std"}).sort_values("bits")
    dense = d[d.method == "dense"].recall_mean.iloc[0]
    for ax, ttl in ((axes[0], "a   LoCoMo: evidence recall@10"),):
        curve_panel(ax, fly, sim, dense, "recall@10", ttl, ann_k=None)
    axes[0].set_ylim(0, 0.52)

    # (b) labeled benchmark: mAP
    flyb = lab[lab.method == "flyhash"].rename(columns={"map_mean": "mean", "map_std": "std"}).sort_values("bits")
    simb = lab[lab.method == "simhash"].rename(columns={"map_mean": "mean", "map_std": "std"}).sort_values("bits")
    curve_panel(axes[1], flyb, simb, lab.dense_map.iloc[0], "mAP", "b   Labeled benchmark: mAP", ann_k=None)
    axes[1].set_ylim(0, 0.52)

    handles = [
        Line2D([0], [0], color=BLUE, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, label="FlyHash, by stored bits (11 bits per active unit)"),
        Line2D([0], [0], color=BLUE, lw=1.4, ls=(0, (1.5, 2)), marker="o", ms=5, markerfacecolor=SURFACE,
               markeredgecolor=BLUE, markeredgewidth=1.4, label="FlyHash, by active bits"),
        Line2D([0], [0], color=ORANGE, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, label="SimHash, by bits"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.075), fontsize=8, handlelength=2.6,
               columnspacing=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_equal_storage.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_locomo_detail(lo, boot, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.3, 1]}, constrained_layout=True)

    # (a) per-category recall@10, mean-centred
    cats = [("1", "multi-hop"), ("2", "temporal"), ("3", "open-domain"), ("4", "single-hop")]
    d = lo[(lo.k == 10) & (lo.centered == True)]  # noqa: E712

    def val(method, bits, cat):
        r = d[(d.method == method) & (d.bits == bits) & (d.category.astype(str) == cat)]
        return float(r.recall_mean.iloc[0])

    series = [("FlyHash, 128 active bits", BLUE, "flyhash", 128),
              ("SimHash, 128 bits", ORANGE_LIGHT, "simhash", 128),
              ("SimHash, 1,024 bits", ORANGE, "simhash", 1024),
              ("dense cosine", "#b9b8b2", "dense", 0)]
    style(a)
    w, gap = 0.19, 0.02
    x0 = np.arange(len(cats))
    for i, (lab, col, m, bt) in enumerate(series):
        xs = x0 + (i - 1.5) * (w + gap)
        ys = [val(m, bt, c) for c, _ in cats]
        bars = a.bar(xs, ys, width=w, color=col, linewidth=0)
        for bar in bars:                      # rounded data ends are approximated by a thin surface-colored cap line
            bar.set_zorder(3)
    a.set_xticks(x0)
    a.set_xticklabels([n for _, n in cats], fontsize=8)
    a.tick_params(axis="x", length=0)
    a.set_ylabel("recall@10")
    a.set_ylim(0, 0.75)
    a.set_title("a   By question type (mean-centred)", loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=8)
    a.legend(handles=[Patch(color=c, label=l) for l, c, _, _ in series], loc="upper left", fontsize=7.6, ncol=2,
             columnspacing=1.0, handlelength=1.1, borderaxespad=0.1)

    # (b) paired bootstrap differences (recall@10, mean-centred, 1,535 questions)
    bt = boot[(boot.k == 10) & (boot.centered == True)]  # noqa: E712
    style(b)
    b.grid(axis="y", visible=False)
    b.grid(axis="x", color=GRID, linewidth=0.8)
    rows = [16, 32, 64, 128]
    ypos = np.arange(len(rows))[::-1]
    for y, k in zip(ypos, rows):
        for dy, other, col in ((0.13, k, BLUE), (-0.13, 4 * k, ORANGE)):
            r = bt[(bt.a == f"flyhash{k}") & (bt.b == f"simhash{other}")]
            if r.empty:
                continue
            r = r.iloc[0]
            b.plot([r.ci_lo, r.ci_hi], [y + dy, y + dy], color=col, linewidth=2, solid_capstyle="round", zorder=3)
            b.plot(r["diff"], y + dy, "o", color=col, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
    b.axvline(0, color=INK2, linewidth=1)
    b.set_yticks(ypos)
    b.set_yticklabels([f"{k} active bits" for k in rows])
    b.tick_params(axis="y", length=0)
    b.set_xlim(-0.06, 0.19)
    b.set_ylim(-0.6, len(rows) + 0.55)
    b.set_xlabel("FlyHash minus SimHash, recall@10 (95% CI)")
    b.set_title("b   Paired bootstrap over questions", loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=8)
    b.text(0.185, -0.55, "FlyHash ahead →", ha="right", va="bottom", fontsize=7.6, color=INK2)
    b.text(-0.055, -0.55, "← SimHash ahead", ha="left", va="bottom", fontsize=7.6, color=INK2)
    b.legend(handles=[Line2D([0], [0], color=BLUE, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, label="vs SimHash, same bits"),
                      Line2D([0], [0], color=ORANGE, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, label="vs SimHash, 4× the bits")],
             loc="upper right", fontsize=7.6, handlelength=1.8, borderaxespad=0.1, ncol=1)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig4_locomo_detail.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", required=True)
    ap.add_argument("--locomo-boot", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("font:", setup_fonts())
    lo = pd.read_csv(a.locomo)
    boot = pd.read_csv(a.locomo_boot)
    lab = pd.read_csv(a.labeled)
    fig_equal_storage(lo, lab, out)
    fig_locomo_detail(lo, boot, out)
    print("wrote", sorted(p.name for p in out.glob("fig[34]_*")))


if __name__ == "__main__":
    main()
