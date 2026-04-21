"""
Created on April 20, 2026.
figure3.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/crossmodal/
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Style
plt.rcParams["font.family"]        = "sans-serif"
plt.rcParams["font.sans-serif"]    = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.linewidth"]     = 0.8
plt.rcParams["axes.labelsize"]     = 10
plt.rcParams["xtick.labelsize"]    = 8.5
plt.rcParams["ytick.labelsize"]    = 8.5
plt.rcParams["legend.fontsize"]    = 9
plt.rcParams["legend.frameon"]     = False
plt.rcParams["axes.grid"]          = False
plt.rcParams["xtick.major.size"]   = 3
plt.rcParams["ytick.major.size"]   = 3
plt.rcParams["xtick.major.width"]  = 0.8
plt.rcParams["ytick.major.width"]  = 0.8
plt.rcParams["hatch.linewidth"]    = 0.8


# Constants
MODEL_ORDER  = ["CLIP", "PubMedCLIP", "BiomedCLIP", "BioViL-T"]
MODEL_COLORS = {
    "BioViL-T":   "#E64B35",
    "BiomedCLIP": "#4DBBD5",
    "PubMedCLIP": "#00A087",
    "CLIP":       "#3C5488",
}
MODEL_MARKERS = {
    "BioViL-T":   "o",
    "BiomedCLIP": "s",
    "PubMedCLIP": "^",
    "CLIP":       "D",
}
MODEL_MS = {
    "BioViL-T":   6.5,
    "BiomedCLIP": 5.5,
    "PubMedCLIP": 5.5,
    "CLIP":       5.0,
}

DATASETS       = ["mimic", "chexpert"]
DATASET_LABELS = {"mimic": "MIMIC-CXR", "chexpert": "CheXpert Plus"}
PANEL_LABELS   = [["a", "b", "c"], ["d", "e", "f"]]
N_EVAL         = 10_000


# Utilities
def parse_cell(cell):
    """Parse 'mean ± std [lo, hi]' -> (mean, std, lo, hi)."""
    cell = str(cell).strip()
    m = re.match(r"([\d.]+)\s*±\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", cell)
    if m:
        return tuple(float(x) for x in m.groups())
    return (float(cell), 0.0, float(cell), float(cell))


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["model_name"]   = df["model_name"].str.strip().replace({"Clip": "CLIP"})
    df["dataset_name"] = df["dataset_name"].str.strip()
    df["variant_name"] = df["variant_name"].str.strip()

    for col in ["recall@1", "recall@5", "recall@10", "mrr"]:
        parsed = df[col].apply(parse_cell)
        df[f"{col}_mean"] = parsed.apply(lambda x: x[0])
        df[f"{col}_std"]  = parsed.apply(lambda x: x[1])
        df[f"{col}_lo"]   = parsed.apply(lambda x: x[2])
        df[f"{col}_hi"]   = parsed.apply(lambda x: x[3])
    return df


def chance_recall1(N):
    return 100.0 / N


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_stats(rand_sub, hard_sub, model):
    r_row = rand_sub[rand_sub["model_name"] == model]
    h_row = hard_sub[hard_sub["model_name"] == model]
    if r_row.empty or h_row.empty:
        return None

    r = r_row.iloc[0]
    h = h_row.iloc[0]
    ch = chance_recall1(N_EVAL)

    return {
        "model": model,
        "r_mean": r["recall@1_mean"],
        "r_std":  r["recall@1_std"],
        "r_hi":   r["recall@1_hi"],
        "h_mean": h["recall@1_mean"],
        "h_std":  h["recall@1_std"],
        "h_hi":   h["recall@1_hi"],
        "drop":   ((r["recall@1_mean"] - h["recall@1_mean"]) / r["recall@1_mean"] * 100.0)
                  if r["recall@1_mean"] > 0 else 0.0,
        "r_fold": r["recall@1_mean"] / ch,
        "r_fold_std": r["recall@1_std"] / ch,
        "h_fold": h["recall@1_mean"] / ch,
        "h_fold_std": h["recall@1_std"] / ch,
    }



# Panel drawing functions
def draw_slopegraph(ax, rand_sub, hard_sub, label, ymax, show_ylabel):
    """Panel a/d: Slopegraph showing Random -> Hard-neg drop."""
    ch = chance_recall1(N_EVAL)

    for model in MODEL_ORDER:
        s = get_stats(rand_sub, hard_sub, model)
        if s is None:
            continue

        c  = MODEL_COLORS[model]
        mk = MODEL_MARKERS[model]
        ms = MODEL_MS[model]

        ax.plot([0.0, 1.0], [s["r_mean"], s["h_mean"]],
                color=c, lw=1.3, alpha=0.75, zorder=2)

        ax.errorbar(0.0, s["r_mean"], yerr=s["r_std"],
                    fmt=mk, color=c, markersize=ms + 1,
                    markerfacecolor="white", markeredgecolor=c,
                    markeredgewidth=1.4,
                    elinewidth=0.8, capsize=2.5, capthick=0.8,
                    zorder=4)

        ax.errorbar(1.0, s["h_mean"], yerr=s["h_std"],
                    fmt=mk, color=c, markersize=ms + 1,
                    markerfacecolor=c, markeredgecolor=c,
                    elinewidth=0.8, capsize=2.5, capthick=0.8,
                    zorder=4)

    ax.axhline(ch, color="#888888", lw=0.9, ls="--", zorder=1)
    ax.text(1.04, ch * 1.12, "chance", fontsize=9,
            color="#888888", va="bottom", ha="left")

    ax.set_yscale("log")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(ch * 0.3, ymax)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Random", "Hard-negative"],
                       rotation=20, ha="right", fontsize=9.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.2f}" if v < 1 else f"{v:.1f}"
    ))
    if show_ylabel:
        ax.set_ylabel("Recall@1 (%)", fontsize=10, labelpad=3)
    ax.text(0.02, 1.06, "Recall@1: random vs. hard-negative",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="left")
    ax.text(-0.22, 1.06, f"$\\mathbf{{{label}}}$",
            transform=ax.transAxes, fontsize=16, va="bottom", ha="left")
    despine(ax)


def draw_relative_drop(ax, rand_sub, hard_sub, label, ymax, show_ylabel):
    x = np.arange(len(MODEL_ORDER))
    width = 0.66

    for i, model in enumerate(MODEL_ORDER):
        s = get_stats(rand_sub, hard_sub, model)
        if s is None:
            continue

        c = MODEL_COLORS[model]
        ax.bar(x[i], s["drop"], width=width, color=c, alpha=0.88, zorder=3)
        ax.text(x[i], s["drop"] + ymax * 0.02, f'{s["drop"]:.1f}%',
                ha="center", va="bottom", fontsize=8.3, color="#333333")

    ax.axhline(100, color="#888888", lw=0.9, ls="--", zorder=2)
    ax.text(len(MODEL_ORDER) - 0.05, 100 + ymax * 0.015,
            "100% (fully explained)",
            fontsize=10, color="#888888", va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
    ax.set_ylim(0, ymax)
    if show_ylabel:
        ax.set_ylabel("Recall@1 reduction (%)", fontsize=10, labelpad=3)
    ax.text(0.02, 1.06, "Relative drop due to hard negatives",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="left")
    ax.text(-0.16, 1.06, f"$\\mathbf{{{label}}}$",
            transform=ax.transAxes, fontsize=16, va="bottom", ha="left")
    despine(ax)


def draw_fold_bars(ax, rand_sub, hard_sub, label, ymax, show_ylabel):
    """Panel c/f: Grouped vertical bars showing fold-above-chance."""
    x = np.arange(len(MODEL_ORDER))
    width = 0.34

    for i, model in enumerate(MODEL_ORDER):
        s = get_stats(rand_sub, hard_sub, model)
        if s is None:
            continue

        c = MODEL_COLORS[model]

        ax.bar(x[i] - width / 2, s["r_fold"], width=width,
               color=c, alpha=0.92, zorder=3)
        ax.errorbar(x[i] - width / 2, s["r_fold"], yerr=s["r_fold_std"],
                    fmt="none", color="black",
                    elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)

        ax.bar(x[i] + width / 2, s["h_fold"], width=width,
               color=c, alpha=0.35, hatch="///",
               edgecolor=c, linewidth=0.6, zorder=3)
        ax.errorbar(x[i] + width / 2, s["h_fold"], yerr=s["h_fold_std"],
                    fmt="none", color="black",
                    elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)

    ax.axhline(1.0, color="#555555", lw=0.9, ls="--", zorder=2)
    ax.text(len(MODEL_ORDER) - 0.05, 1.0 + ymax * 0.015,
            "1$\\times$ chance",
            fontsize=9, color="#555555", va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
    ax.set_ylim(0, ymax)
    if show_ylabel:
        ax.set_ylabel("Recall@1 / chance (fold)", fontsize=10, labelpad=3)
    ax.text(0.02, 1.06, "Residual fold above chance",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="left")
    ax.text(-0.16, 1.06, f"$\\mathbf{{{label}}}$",
            transform=ax.transAxes, fontsize=16, va="bottom", ha="left")
    despine(ax)


def draw_legend(fig):
    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[m],
               marker=MODEL_MARKERS[m], markersize=MODEL_MS[m],
               linewidth=1.4, label=m)
        for m in MODEL_ORDER
    ]
    setting_handles = [
        Line2D([0], [0], color="black", marker="o", markersize=7,
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.3, linewidth=0,
               label="Random (open marker)"),
        Line2D([0], [0], color="black", marker="o", markersize=7,
               markerfacecolor="black", linewidth=0,
               label="Hard-negative (filled marker)"),
        Patch(facecolor="#555555", alpha=0.85,
              label="Random bar (panels c, f)"),
        Patch(facecolor="#888888", alpha=0.40, hatch="///",
              edgecolor="#555555",
              label="Hard-negative bar (panels c, f)"),
    ]
    fig.legend(
        handles=model_handles + setting_handles,
        loc="upper center", ncol=8,
        fontsize=10, frameon=False,
        bbox_to_anchor=(0.53, 0.985),
        handlelength=1.8, handletextpad=0.45, columnspacing=0.95,
    )


# Main
def make_figure(csv_path, out_path):
    df = load_data(csv_path)

    df_rand = df[
        (df["variant_name"] == "off_shelf") &
        (df["setting"] == "A") &
        (df["N"] == N_EVAL)
    ].copy()

    df_hard = df[
        (df["variant_name"] == "off_shelf") &
        (df["setting"] == "B") &
        (df["N"] == N_EVAL)
    ].copy()

    fig = plt.figure(figsize=(14.2, 8.8))
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.60, wspace=0.34,
        left=0.085, right=0.995,
        top=0.875, bottom=0.10,
    )

    ch = chance_recall1(N_EVAL)
    all_r1_hi = pd.concat([df_rand, df_hard])["recall@1_hi"]
    slope_ymax = max(all_r1_hi.max() * 1.8, ch * 20)

    all_fold_hi = pd.concat([df_rand, df_hard])["recall@1_hi"] / ch
    fold_ymax = all_fold_hi.max() * 1.18

    drop_ymax = 112

    for ri, dataset in enumerate(DATASETS):
        rand_sub = df_rand[df_rand["dataset_name"] == dataset].copy()
        hard_sub = df_hard[df_hard["dataset_name"] == dataset].copy()
        labels   = PANEL_LABELS[ri]

        ax_slope = fig.add_subplot(gs[ri, 0])
        ax_drop  = fig.add_subplot(gs[ri, 1])
        ax_fold  = fig.add_subplot(gs[ri, 2])

        draw_slopegraph(ax_slope, rand_sub, hard_sub,
                        labels[0], slope_ymax,
                        show_ylabel=(ri == 0))
        draw_relative_drop(ax_drop, rand_sub, hard_sub,
                           labels[1], drop_ymax,
                           show_ylabel=(ri == 0))
        draw_fold_bars(ax_fold, rand_sub, hard_sub,
                       labels[2], fold_ymax,
                       show_ylabel=(ri == 0))

    for yi, dataset in zip([0.73, 0.28], DATASETS):
        fig.text(0.01, yi, DATASET_LABELS[dataset],
                 fontsize=11.5, fontweight="bold",
                 va="center", ha="left", rotation=90)

    draw_legend(fig)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")
    plt.close(fig)


