"""
Created on April 20, 2026.
figure5_utility.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/crossmodal/
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch



# Style
plt.rcParams["font.family"]       = "sans-serif"
plt.rcParams["font.sans-serif"]   = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.linewidth"]    = 0.9
plt.rcParams["axes.labelsize"]    = 12
plt.rcParams["xtick.labelsize"]   = 10
plt.rcParams["ytick.labelsize"]   = 10
plt.rcParams["legend.fontsize"]   = 11
plt.rcParams["legend.frameon"]    = False
plt.rcParams["axes.grid"]         = False
plt.rcParams["xtick.major.size"]  = 3.5
plt.rcParams["ytick.major.size"]  = 3.5
plt.rcParams["xtick.major.width"] = 0.9
plt.rcParams["ytick.major.width"] = 0.9

# Colors
COLOR_OFFSHELF = "#BBBBBB"
COLOR_DP       = "#4477AA"
COLOR_UP       = "#2E8B57"
COLOR_DOWN     = "#C04040"
COLOR_NEUTRAL  = "#B0B0B0"

# Paths
CSV_PRIVACY = "/PATH.csv"
CSV_DP = "/PATH.csv"
CSV_OFFSHELF = "/PATH.csv"
OUT_PATH     = "mitigation_utility.png"

LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged_cardiomediastinum", "fracture", "lung_lesion", "lung_opacity",
    "no_finding", "pleural_effusion", "pleural_other", "pneumonia",
    "pneumothorax", "support_devices",
]
LABEL_PRETTY = {
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "edema": "Edema",
    "enlarged_cardiomediastinum": "Enlarged cardiomed.",
    "fracture": "Fracture",
    "lung_lesion": "Lung lesion",
    "lung_opacity": "Lung opacity",
    "no_finding": "No finding",
    "pleural_effusion": "Pleural effusion",
    "pleural_other": "Pleural other",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "support_devices": "Support devices",
}

# Helpers
def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, letter, x=-0.16):
    ax.text(x, 1.06, f"$\\mathbf{{{letter}}}$",
            transform=ax.transAxes, fontsize=18, va="bottom", ha="left")


def panel_title(ax, title, x=0.02):
    ax.text(x, 1.06, title,
            transform=ax.transAxes, fontsize=12, va="bottom", ha="left")


def parse_cell(cell):
    cell = str(cell).strip()
    m = re.match(r"([\d.]+)\s*±\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", cell)
    if m:
        return tuple(float(x) for x in m.groups())
    return (float(cell), 0.0, float(cell), float(cell))


def load_privacy_value(csv_path):
    df = pd.read_csv(csv_path)
    df["model_name"]   = df["model_name"].str.strip().replace({"Clip": "CLIP"})
    df["variant_name"] = df["variant_name"].str.strip()
    df["dataset_name"] = df["dataset_name"].str.strip()

    def get(variant):
        row = df[
            (df["dataset_name"] == "mimic") &
            (df["model_name"] == "BioViL-T") &
            (df["variant_name"] == variant) &
            (df["setting"] == "A") &
            (df["N"] == 10000)
        ]
        return parse_cell(row["recall@1"].values[0])[0]

    return get("off_shelf"), get("dp_head")


# Panel a
def draw_tradeoff(ax, priv_rem, util_rem, priv_red, util_red):
    categories  = ["Image utility\n(macro AUROC)",
                   "Privacy risk\n(Recall@1, $N=10{,}000$)"]
    y_positions = np.arange(len(categories))

    values_remain = [util_rem, priv_rem]
    values_lost   = [100 - v for v in values_remain]

    bar_h = 0.42
    ax.barh(y_positions, values_remain, height=bar_h,
            color=COLOR_DP, alpha=0.88, zorder=3)
    ax.barh(y_positions, values_lost, left=values_remain, height=bar_h,
            color=COLOR_OFFSHELF, alpha=0.55, zorder=3)

    ax.axvline(100, color="#555555", lw=0.9, ls=":", zorder=2)

    for i, (rem, lost) in enumerate(zip(values_remain, values_lost)):
        y = y_positions[i]
        rel = util_red if i == 0 else priv_red

        if lost > 5:
            ax.text(rem + lost / 2, y,
                    f"$-${rel:.1f}%",
                    ha="center", va="center", fontsize=11, color="#333333")
        else:
            ax.text(102, y, f"$-${rel:.2f}%",
                    ha="left", va="center", fontsize=10.5, color="#555555")

        if rem > 15:
            ax.text(rem - 1.5, y, f"{rem:.1f}%",
                    ha="right", va="center", fontsize=10, color="white")
        else:
            ax.text(rem + 1.5, y, f"{rem:.1f}%",
                    ha="left", va="center", fontsize=10, color=COLOR_DP)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Relative to off-the-shelf (%)", fontsize=11, labelpad=4)

    panel_label(ax, "a", x=-0.32)
    panel_title(ax, "Privacy vs. utility change under DP", x=-0.02)
    despine(ax)


# Panel b
def draw_macro_deltas(ax, off_df, dp_df):
    metrics = ["macro_auroc", "macro_accuracy",
               "macro_sensitivity", "macro_specificity"]
    labels  = ["AUROC", "Accuracy", "Sensitivity", "Specificity"]

    deltas = [(dp_df[m].values - off_df[m].values) * 100.0 for m in metrics]
    positions = np.arange(len(metrics))

    vp = ax.violinplot(deltas, positions=positions, widths=0.75,
                       showmeans=False, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(COLOR_DP)
        body.set_edgecolor(COLOR_DP)
        body.set_alpha(0.35)

    for i, d in enumerate(deltas):
        median = np.median(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        ax.plot(positions[i], median, marker="o", color=COLOR_DP,
                markersize=7, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        ax.plot([positions[i], positions[i]], [lo, hi],
                color=COLOR_DP, lw=1.8, zorder=4)
        for y in [lo, hi]:
            ax.plot([positions[i] - 0.10, positions[i] + 0.10], [y, y],
                    color=COLOR_DP, lw=1.4, zorder=4)

        is_sig = (lo > 0) or (hi < 0)
        y_star = hi + abs(hi - lo) * 0.30 + 0.03
        ax.text(positions[i], y_star,
                "$^{\\ast}$" if is_sig else "",
                ha="center", va="bottom",
                fontsize=14 if is_sig else 10,
                color=COLOR_DP if is_sig else "#777777")

    ax.axhline(0, color="#333333", lw=0.9, ls="-", zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("DP $-$ off-the-shelf (pp)", fontsize=11, labelpad=4)

    panel_label(ax, "b", x=-0.14)
    panel_title(ax, "Paired bootstrap deltas (macro metrics)", x=0.02)
    despine(ax)


# Panel c
def draw_auroc_delta_bars(ax, off_df, dp_df, labels):
    deltas, cis = {}, {}
    for lb in labels:
        d = (dp_df[f"{lb}_auroc"].values - off_df[f"{lb}_auroc"].values) * 100
        deltas[lb] = d.mean()
        cis[lb]    = tuple(np.percentile(d, [2.5, 97.5]))

    sorted_labels = sorted(labels, key=lambda lb: deltas[lb], reverse=True)
    y_positions   = np.arange(len(sorted_labels))[::-1]

    for i, lb in enumerate(sorted_labels):
        y = y_positions[i]
        d = deltas[lb]
        lo, hi = cis[lb]

        if lo > 0:
            c = COLOR_UP
        elif hi < 0:
            c = COLOR_DOWN
        else:
            c = COLOR_NEUTRAL

        ax.barh(y, d, height=0.65, color=c, alpha=0.85, zorder=3)
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]],
                    fmt="none", color="#333333",
                    elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)

    ax.axvline(0, color="#333333", lw=0.9, zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([LABEL_PRETTY[lb] for lb in sorted_labels], fontsize=10)

    all_abs = [abs(v) for v in deltas.values()] + \
              [abs(v) for ci in cis.values() for v in ci]
    xmax = max(all_abs) * 1.2
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("$\\Delta$ AUROC (pp, DP $-$ off-the-shelf)",
                  fontsize=11, labelpad=4)
    ax.set_ylim(-0.5, len(sorted_labels) - 0.5)

    panel_label(ax, "c", x=-0.32)
    panel_title(ax, "Per-label change in AUROC", x=-0.02)
    despine(ax)


# Panel d: per-label heatmap
def draw_delta_heatmap(ax, off_df, dp_df, labels):
    metric_cols = ["auroc", "accuracy", "sensitivity", "specificity"]
    metric_lbls = ["AUROC", "Accuracy", "Sensitivity", "Specificity"]

    n_rows = len(labels)
    n_cols = len(metric_cols)

    delta_matrix = np.zeros((n_rows, n_cols))
    sig_matrix   = np.zeros((n_rows, n_cols), dtype=bool)

    for i, lb in enumerate(labels):
        for j, m in enumerate(metric_cols):
            d = (dp_df[f"{lb}_{m}"].values - off_df[f"{lb}_{m}"].values) * 100
            delta_matrix[i, j] = d.mean()
            lo, hi = np.percentile(d, [2.5, 97.5])
            sig_matrix[i, j] = (lo > 0) or (hi < 0)

    order = sorted(range(n_rows),
                   key=lambda k: abs(delta_matrix[k, 0]), reverse=True)
    delta_matrix = delta_matrix[order]
    sig_matrix   = sig_matrix[order]
    sorted_labels = [labels[k] for k in order]

    vmax = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(delta_matrix, aspect="auto",
                   cmap=cmap, norm=norm, interpolation="nearest")

    for i in range(n_rows):
        for j in range(n_cols):
            # Grey out non-significant cells
            if not sig_matrix[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="white", alpha=0.65, zorder=2,
                    edgecolor="none"))
            val = delta_matrix[i, j]
            txt_color = "white" if abs(val) > vmax * 0.55 and sig_matrix[i, j] else "#333333"
            ax.text(j, i, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=9.5, color=txt_color, zorder=3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(metric_lbls, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([LABEL_PRETTY[lb] for lb in sorted_labels], fontsize=10)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    panel_label(ax, "d", x=-0.32)
    panel_title(ax, "Per-label change across metrics (pp)", x=-0.02)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03, aspect=22)
    cbar.set_label("$\\Delta$ (pp)", fontsize=10, labelpad=4)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)


# Main
def make_figure():
    off_df = pd.read_csv(CSV_OFFSHELF)
    dp_df  = pd.read_csv(CSV_DP)

    privacy_off, privacy_dp = load_privacy_value(CSV_PRIVACY)
    priv_rem = privacy_dp / privacy_off * 100.0
    priv_red = 100.0 - priv_rem

    util_off = off_df["macro_auroc"].mean() * 100.0
    util_dp  = dp_df["macro_auroc"].mean()  * 100.0
    util_rem = util_dp / util_off * 100.0
    util_red = 100.0 - util_rem

    fig = plt.figure(figsize=(14, 11.5))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.55, wspace=0.50,
        left=0.13, right=0.99,
        top=0.86, bottom=0.07,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_tradeoff(ax_a, priv_rem, util_rem, priv_red, util_red)
    draw_macro_deltas(ax_b, off_df, dp_df)
    draw_auroc_delta_bars(ax_c, off_df, dp_df, LABELS)
    draw_delta_heatmap(ax_d, off_df, dp_df, LABELS)

    handles = [
        Patch(facecolor=COLOR_DP,       alpha=0.88, label="Panel a: DP head"),
        Patch(facecolor=COLOR_OFFSHELF, alpha=0.55, label="Panel a: Lost under DP"),
        # Line2D([0], [0], color=COLOR_UP,      lw=3.0,
        #        label="Panel c: Increased 95% CI"),
        Line2D([0], [0], color=COLOR_DOWN,    lw=3.0,
               label="Panel c: Decreased 95% CI"),
        Line2D([0], [0], color=COLOR_NEUTRAL, lw=3.0,
               label="Panel c: Non-significant change"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center", ncol=5,
        fontsize=12, frameon=False,
        bbox_to_anchor=(0.53, 0.955),
        handlelength=2.0, handletextpad=0.55, columnspacing=1.6,
    )

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PATH}")
    plt.close(fig)




if __name__ == "__main__":
    make_figure()
