"""
make_figures.py -- regenerates the README figures in docs/images/.

  pipeline.png      How FlyHash turns a dense vector into a sparse binary code
                    (drawn from a real FlyHash run on a random vector).
  results.png       mAP vs. bit budget, FlyHash vs. SimHash, five datasets, plus
                    FlyHash's relative advantage. Numbers are the paper's
                    Table "main results" and labeled agent-memory table (10 seeds).
  agent_memory.png  Architecture of the FlyHash-backed agent memory.

Usage (from the repo root):   python scripts/make_figures.py
Requires: matplotlib (pip install matplotlib)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)

FLY, SIM, INK, MUTED, FILL = "#0072B2", "#D55E00", "#222222", "#6b7280", "#eef3f8"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


# ---------------------------------------------------------------- pipeline
def fig_pipeline():
    from flyhash import FlyHash
    rng = np.random.default_rng(3)
    d, m, k = 32, 200, 10
    x = rng.standard_normal(d).astype(np.float32)
    fh = FlyHash(input_dim=d, hash_dim=m, sparsity=0.10, wta_percent=k / m, seed=1)
    code = np.asarray(fh.hash(x)).astype(int).ravel()
    W = np.asarray(getattr(fh, "projection", None)) if hasattr(fh, "projection") else None
    act = (W @ x) if W is not None and W.shape == (m, d) else rng.standard_normal(m)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.6, 1.6]})
    ax[0].bar(range(d), x, color=MUTED, width=0.8)
    ax[0].set_title("1  Dense embedding\n(e.g. 32 of 384 dims shown)", fontsize=10)
    ax[0].set_yticks([]); ax[0].set_xticks([])

    ax[1].bar(range(m), act, color=FLY, alpha=0.55, width=1.0)
    ax[1].set_title("2  Sparse random expansion\neach of 2,000 units samples ~10% of inputs", fontsize=10)
    ax[1].set_yticks([]); ax[1].set_xticks([])

    colors = np.where(code > 0, FLY, "#d5dbe3")
    ax[2].bar(range(m), np.where(code > 0, 1.0, 0.07), color=colors, width=1.0)
    ax[2].set_title(f"3  Winner-take-all\nkeep the top k={k} units: a sparse binary code", fontsize=10)
    ax[2].set_yticks([]); ax[2].set_xticks([])
    for a in ax:
        a.spines["left"].set_visible(False)
    fig.text(0.5, -0.02, "Similar inputs light up overlapping units, so Hamming distance "
             "on the code ranks neighbours. SimHash instead keeps one sign bit per hyperplane.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------- results
K = [16, 32, 64, 128, 256]
DATA = {  # (FlyHash mean, std, SimHash mean, std) per k  -- paper tables, 10 seeds
 "GloVe words (mAP@100)": ([.196,.314,.449,.585,.705],[.007,.009,.008,.006,.006],[.047,.096,.185,.316,.469],[.006,.009,.008,.006,.007]),
 "MiniLM sentences (mAP@20)": ([.310,.408,.537,.656,.753],[.013,.018,.030,.016,.013],[.214,.266,.355,.466,.587],[.026,.022,.024,.027,.016]),
 "Digits images (mAP@50)": ([.456,.593,.699,.748,.806],[.015,.010,.011,.005,.007],[.119,.236,.397,.535,.664],[.031,.039,.023,.021,.009]),
 "MNIST images (mAP@100)": ([.276,.382,.469,.527,.563],[.009,.010,.011,.008,.004],[.064,.133,.237,.351,.450],[.014,.012,.014,.009,.010]),
 "Agent memory, labeled (mAP)": ([.195,.271,.324,.365,.392],[.014,.017,.012,.009,.009],[.038,.105,.212,.298,.345],[.010,.011,.016,.011,.011]),
}

def fig_results():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    axes = axes.ravel()
    for a, (name, (f, fs, s, ss)) in zip(axes, DATA.items()):
        a.errorbar(K, f, yerr=fs, color=FLY, marker="o", lw=2, capsize=3, label="FlyHash")
        a.errorbar(K, s, yerr=ss, color=SIM, marker="s", lw=2, capsize=3, label="SimHash")
        a.set_xscale("log", base=2); a.set_xticks(K); a.set_xticklabels(K)
        a.set_title(name, fontsize=10.5, loc="left"); a.set_ylim(0, None)
        a.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    ax = axes[5]
    styles = [("#0072B2", "o"), ("#009E73", "s"), ("#CC79A7", "^"), ("#E69F00", "D"), ("#222222", "v")]
    for (name, (f, _, s, _)), (c, mk) in zip(DATA.items(), styles):
        ax.plot(K, np.array(f) / np.array(s), "-", marker=mk, ms=5, lw=1.8,
                color=c, label=name.split(" (")[0])
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(K); ax.set_xticklabels(K)
    ax.set_title("FlyHash / SimHash (x better)", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=7.5); ax.grid(axis="y", alpha=0.25)
    for a in axes[3:]:
        a.set_xlabel("active bits k")
    for a in axes[[0, 3]]:
        a.set_ylabel("mAP")
    fig.suptitle("Retrieval quality at equal bit budget (mean ± std, 10 seeds)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "results.png"), dpi=160, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ architecture
def box(ax, xy, w, h, text, fc=FILL, ec=MUTED, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9.5,
            fontweight="bold" if bold else "normal")

def arrow(ax, a, b, text=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=MUTED))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.13, text, ha="center", fontsize=8.5, color=MUTED)

def fig_agent():
    fig, ax = plt.subplots(figsize=(12, 4.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(0.1, 3.95, "Write path", fontsize=10, fontweight="bold", color=FLY)
    box(ax, (0.1, 2.7), 2.0, 0.9, "Conversation\n\"I'm allergic\nto peanuts\"")
    box(ax, (2.9, 2.7), 2.0, 0.9, "Sentence embedding\n(MiniLM, 384-d)")
    box(ax, (5.7, 2.7), 2.1, 0.9, "FlyHash\nsparse binary code", fc="#dcebf7", ec=FLY, bold=True)
    box(ax, (8.6, 2.7), 3.2, 0.9, "Memory store\ntext + k active-bit indices")
    ax.text(0.1, 3.68, "agent tool: save_memory", fontsize=8.5, color=MUTED); arrow(ax, (2.1, 3.15), (2.9, 3.15)); arrow(ax, (4.9, 3.15), (5.7, 3.15))
    arrow(ax, (7.8, 3.15), (8.6, 3.15))
    ax.text(0.1, 1.85, "Read path", fontsize=10, fontweight="bold", color=SIM)
    box(ax, (0.1, 0.5), 2.0, 0.9, "Later question\n\"What should I\navoid eating?\"")
    box(ax, (2.9, 0.5), 2.0, 0.9, "Embed + FlyHash\n(same projection)", fc="#dcebf7", ec=FLY)
    box(ax, (5.7, 0.5), 2.1, 0.9, "Rank by overlap\n(Hamming, top-k)")
    box(ax, (8.6, 0.5), 3.2, 0.9, "Recalled memories\n-> LLM context -> answer")
    ax.text(0.1, 1.48, "agent tool: recall_memory", fontsize=8.5, color=MUTED); arrow(ax, (2.1, 0.95), (2.9, 0.95)); arrow(ax, (4.9, 0.95), (5.7, 0.95))
    arrow(ax, (7.8, 0.95), (8.6, 0.95))
    arrow(ax, (10.2, 2.7), (7.0, 1.4)); 
    fig.savefig(os.path.join(OUT, "agent_memory.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); fig_results(); fig_agent()
    print("wrote", sorted(os.listdir(OUT)))
