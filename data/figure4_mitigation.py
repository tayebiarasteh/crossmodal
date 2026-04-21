"""
Created on April 20, 2026.
figure4_mitigation.py

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
plt.rcParams["font.family"]       = "sans-serif"
plt.rcParams["font.sans-serif"]   = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.linewidth"]    = 0.8
plt.rcParams["axes.labelsize"]    = 10
plt.rcParams["xtick.labelsize"]   = 8.5
plt.rcParams["ytick.labelsize"]   = 8.5
plt.rcParams["legend.fontsize"]   = 9
plt.rcParams["legend.frameon"]    = False
plt.rcParams["axes.grid"]         = False
plt.rcParams["xtick.major.size"]  = 3
plt.rcParams["ytick.major.size"]  = 3
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8


# Constants
VARIANT_ORDER  = ["off_shelf", "dp_head", "entropy_head"]
VARIANT_LABELS = {"off_shelf": "Off-the-shelf", "dp_head": "DP head", "entropy_head": "Entropy head"}
VARIANT_COLORS = {"off_shelf": "#333333",  "dp_head": "#4477AA",  "entropy_head": "#EE6677"}
VARIANT_MARKERS = {"off_shelf": "o",       "dp_head": "s",        "entropy_head": "^"}
VARIANT_MS     = {"off_shelf": 5.5,        "dp_head": 5.5,        "entropy_head": 5.5}

DATASETS       = ["mimic", "chexpert"]
DATASET_LABELS = {"mimic": "MIMIC-CXR", "chexpert": "CheXpert Plus"}
FULL_N         = {"mimic": None, "chexpert": None}   # filled from data

N_EVAL = 10_000


# Utilities
def parse_cell(cell):
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


def x_tick_labels(pools, full_N):
    out = []
    for p in pools:
        if p == full_N:
            out.append(f"{p:,}\n(full)")
        elif p >= 1000:
            out.append(f"{p // 1000:,}K")
        else:
            out.append(str(p))
    return out


def panel_label(ax, letter, has_wide_ylabel=True):
    x = -0.22 if has_wide_ylabel else -0.14
    ax.text(x, 1.06, f"$\\mathbf{{{letter}}}$",
            transform=ax.transAxes, fontsize=16, va="bottom", ha="left")


def panel_title(ax, title):
    ax.text(0.02, 1.06, title,
            transform=ax.transAxes, fontsize=10, va="bottom", ha="left")



# Panel a/d/b/e: trajectory line plots (log-log)
def draw_trajectory(ax, df_settingA, dataset, metric_col, metric_label,
                    full_N, show_ylabel, shared_ylim, letter):
    pools = []
    for variant in VARIANT_ORDER:
        sub = df_settingA[
            (df_settingA["dataset_name"] == dataset) &
            (df_settingA["variant_name"] == variant)
        ].sort_values("N")
        if sub.empty:
            continue
        xs   = sub["N"].values
        ys   = sub[f"{metric_col}_mean"].values
        stds = sub[f"{metric_col}_std"].values
        pools = sorted(set(pools) | set(xs.tolist()))

        ax.errorbar(xs, ys, yerr=stds,
                    color=VARIANT_COLORS[variant],
                    marker=VARIANT_MARKERS[variant],
                    markersize=VARIANT_MS[variant],
                    linewidth=1.4, capsize=2.2,
                    capthick=0.9, elinewidth=0.7,
                    zorder=3 + VARIANT_ORDER.index(variant),
                    label=VARIANT_LABELS[variant])

    pools = sorted(pools)
    # Chance line
    ch_vals = [chance_recall1(N) for N in pools]
    ax.plot(pools, ch_vals, color="#888888", lw=0.9, ls="--", zorder=2)
    ax.text(pools[0] * 1.1, ch_vals[0] * 1.15, "chance",
            fontsize=9.5, color="#888888", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*shared_ylim)
    ax.set_xticks(pools)
    ax.set_xticklabels(x_tick_labels(pools, full_N), fontsize=8)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.1f}" if v >= 1 else f"{v:.2f}"
    ))
    ax.set_xlabel("Candidate-pool size $N$", fontsize=10, labelpad=3)
    if show_ylabel:
        ax.set_ylabel(f"{metric_label} (%)", fontsize=10, labelpad=3)
    panel_label(ax, letter, has_wide_ylabel=show_ylabel)
    panel_title(ax, metric_label)
    despine(ax)


# Panel c/f: % change from off-the-shelf across pool sizes
def draw_pct_change_lines(ax, df_settingA, dataset, full_N,
                          show_ylabel, shared_ylim, letter):
    """
    Line plot: % change in Recall@1 from off-the-shelf for DP and entropy.
    """
    ref = df_settingA[
        (df_settingA["dataset_name"] == dataset) &
        (df_settingA["variant_name"] == "off_shelf")
    ].sort_values("N").set_index("N")

    for variant in ["dp_head", "entropy_head"]:
        sub = df_settingA[
            (df_settingA["dataset_name"] == dataset) &
            (df_settingA["variant_name"] == variant)
        ].sort_values("N")
        if sub.empty:
            continue

        xs, ys, errs = [], [], []
        for _, row in sub.iterrows():
            N = row["N"]
            if N not in ref.index:
                continue
            ref_m = ref.loc[N, "recall@1_mean"]
            ref_s = ref.loc[N, "recall@1_std"]
            v_m   = row["recall@1_mean"]
            v_s   = row["recall@1_std"]
            if ref_m == 0:
                continue
            pct = (v_m - ref_m) / ref_m * 100.0
            # Error propagation
            pct_err = 100.0 * np.sqrt(
                (v_s / ref_m) ** 2 + (v_m * ref_s / ref_m ** 2) ** 2
            )
            xs.append(N)
            ys.append(pct)
            errs.append(pct_err)

        ax.errorbar(xs, ys, yerr=errs,
                    color=VARIANT_COLORS[variant],
                    marker=VARIANT_MARKERS[variant],
                    markersize=VARIANT_MS[variant],
                    linewidth=1.4, capsize=2.2,
                    capthick=0.9, elinewidth=0.7,
                    zorder=3, label=VARIANT_LABELS[variant])

    pools = sorted(ref.index.tolist())
    ax.axhline(0, color="#333333", lw=0.9, ls="-", zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(pools)
    ax.set_xticklabels(x_tick_labels(pools, full_N), fontsize=8)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(*shared_ylim)
    ax.set_xlabel("Candidate-pool size $N$", fontsize=10, labelpad=3)
    if show_ylabel:
        ax.set_ylabel("$\\Delta$Recall@1 vs. off-the-shelf (%)", fontsize=10, labelpad=3)
    panel_label(ax, letter, has_wide_ylabel=show_ylabel)
    panel_title(ax, "$\\Delta$Recall@1 vs. off-the-shelf")
    despine(ax)


# Panel g/h: fold-above-chance grouped bars at N=10K, hard-neg
def draw_hardneg_fold_bars(ax, df_B, df_A, dataset, show_ylabel, letter, title):
    """
    Grouped horizontal bars: fold-above-chance for the 3 variants.
    Solid = random (setting A), hatched = hard-neg (setting B).
    """
    ch    = chance_recall1(N_EVAL)
    bar_h = 0.22
    gap   = 0.05
    n     = len(VARIANT_ORDER)

    pos_rand = np.arange(n) * (3 * bar_h + 3 * gap)
    pos_hard = pos_rand + bar_h + gap

    for i, variant in enumerate(VARIANT_ORDER):
        # Random (setting A)
        r_row = df_A[
            (df_A["dataset_name"] == dataset) &
            (df_A["variant_name"] == variant) &
            (df_A["N"] == N_EVAL)
        ]
        # Hard-neg (setting B)
        h_row = df_B[
            (df_B["dataset_name"] == dataset) &
            (df_B["variant_name"] == variant) &
            (df_B["N"] == N_EVAL)
        ]

        c = VARIANT_COLORS[variant]

        if not r_row.empty:
            r_fold = r_row["recall@1_mean"].values[0] / ch
            r_std  = r_row["recall@1_std"].values[0]  / ch
            ax.barh(pos_rand[i], r_fold, height=bar_h,
                    color=c, alpha=0.90, zorder=3)
            ax.errorbar(r_fold, pos_rand[i], xerr=r_std,
                        fmt="none", color="black",
                        elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)

        if not h_row.empty:
            h_fold = h_row["recall@1_mean"].values[0] / ch
            h_std  = h_row["recall@1_std"].values[0]  / ch
            ax.barh(pos_hard[i], h_fold, height=bar_h,
                    color=c, alpha=0.45, hatch="///",
                    edgecolor=c, linewidth=0.5, zorder=3)
            ax.errorbar(h_fold, pos_hard[i], xerr=h_std,
                        fmt="none", color="black",
                        elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)

    ax.axvline(1.0, color="#555555", lw=0.9, ls="--", zorder=2)
    ax.text(1.05, pos_hard[-1] + bar_h, "1$\\times$\nchance",
            fontsize=8.5, color="#555555", va="bottom", ha="left")

    ytick_pos = (pos_rand + pos_hard) / 2
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], fontsize=8.5)
    ax.set_ylim(-bar_h * 1.5, pos_hard[-1] + bar_h * 2)
    ax.set_xlabel("Recall@1 / chance (fold)", fontsize=10, labelpad=3)
    panel_label(ax, letter, has_wide_ylabel=show_ylabel)
    panel_title(ax, title)
    despine(ax)


# Panel i: % change under hard-neg, both datasets, horizontal bars
def draw_hardneg_pct_change(ax, df_A, df_B, letter):
    variants = ["dp_head", "entropy_head"]
    n        = len(variants)
    bar_h    = 0.28
    gap      = 0.08
    ds_gap   = 0.18

    pos_mimic  = np.arange(n) * (2 * bar_h + gap + ds_gap)
    pos_chex   = pos_mimic + bar_h + gap

    for i, variant in enumerate(variants):
        for j, (dataset, pos_arr, hatch, alpha) in enumerate([
            ("mimic",    pos_mimic, "",    0.90),
            ("chexpert", pos_chex,  "///", 0.45),
        ]):
            ref_row = df_B[
                (df_B["dataset_name"] == dataset) &
                (df_B["variant_name"] == "off_shelf") &
                (df_B["N"] == N_EVAL)
            ]
            var_row = df_B[
                (df_B["dataset_name"] == dataset) &
                (df_B["variant_name"] == variant) &
                (df_B["N"] == N_EVAL)
            ]

            if ref_row.empty or var_row.empty:
                continue

            ref_m = ref_row["recall@1_mean"].values[0]
            ref_s = ref_row["recall@1_std"].values[0]
            v_m   = var_row["recall@1_mean"].values[0]
            v_s   = var_row["recall@1_std"].values[0]

            if ref_m == 0:
                continue

            pct = (v_m - ref_m) / ref_m * 100.0
            pct_err = 100.0 * np.sqrt(
                (v_s / ref_m) ** 2 + (v_m * ref_s / ref_m ** 2) ** 2
            )

            c = VARIANT_COLORS[variant]
            ax.barh(pos_arr[i], pct, height=bar_h,
                    color=c, alpha=alpha, hatch=hatch,
                    edgecolor=c, linewidth=0.5 if hatch else 0, zorder=3)
            ax.errorbar(pct, pos_arr[i], xerr=pct_err,
                        fmt="none", color="black",
                        elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)

    ax.axvline(0, color="#333333", lw=0.9, ls="-", zorder=2)

    ytick_pos = (pos_mimic + pos_chex) / 2
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([VARIANT_LABELS[v] for v in variants], fontsize=8.5)
    ax.set_ylim(-bar_h * 1.5, pos_chex[-1] + bar_h * 2)
    ax.set_xlabel("$\\Delta$Recall@1 vs. off-the-shelf (%)",
                  fontsize=10, labelpad=3)
    panel_label(ax, letter, has_wide_ylabel=False)
    panel_title(ax, "$\\Delta$Recall@1 under hard negatives")
    despine(ax)

# Legend
def draw_legend(fig):
    variant_handles = [
        Line2D([0], [0], color=VARIANT_COLORS[v],
               marker=VARIANT_MARKERS[v], markersize=VARIANT_MS[v],
               linewidth=1.4, label=VARIANT_LABELS[v])
        for v in VARIANT_ORDER
    ]
    extra_handles = [
        Line2D([0], [0], color="#888888", lw=0.9, ls="--", label="Chance"),
        Patch(facecolor="#888888", alpha=0.85, label="Random (solid bars)"),
        Patch(facecolor="#888888", alpha=0.40, hatch="///",
              edgecolor="#888888", label="Hard-neg (hatched bars)"),
    ]
    fig.legend(
        handles=variant_handles + extra_handles,
        loc="upper center", ncol=6,
        fontsize=12, frameon=False,
        bbox_to_anchor=(0.53, 0.975),
        handlelength=1.8, handletextpad=0.45, columnspacing=1.0,
    )


# Main
def make_figure(csv_path, out_path):
    df = load_data(csv_path)

    df_A = df[df["setting"] == "A"].copy()  # random-candidate
    df_B = df[df["setting"] == "B"].copy()  # hard-negative

    # Only BioViL-T variants here
    df_A = df_A[df_A["model_name"] == "BioViL-T"]
    df_B = df_B[df_B["model_name"] == "BioViL-T"]

    # Full N per dataset
    for ds in DATASETS:
        FULL_N[ds] = int(df_A[df_A["dataset_name"] == ds]["N"].max())

    # Shared y-limits: compute across both datasets per metric
    def ylim_trajectory(metric_col):
        vals = df_A[df_A["variant_name"].isin(VARIANT_ORDER)][f"{metric_col}_mean"]
        stds = df_A[df_A["variant_name"].isin(VARIANT_ORDER)][f"{metric_col}_std"]
        top  = (vals + stds).max() * 2.0
        bot  = max((vals - stds).min() * 0.4, 1e-4)
        return (bot, top)

    ylim_r1  = ylim_trajectory("recall@1")
    ylim_mrr = ylim_trajectory("mrr")

    pct_changes = []
    for ds in DATASETS:
        ref = df_A[(df_A["dataset_name"] == ds) &
                   (df_A["variant_name"] == "off_shelf")].set_index("N")
        for variant in ["dp_head", "entropy_head"]:
            sub = df_A[(df_A["dataset_name"] == ds) &
                       (df_A["variant_name"] == variant)]
            for _, row in sub.iterrows():
                N = row["N"]
                if N in ref.index and ref.loc[N, "recall@1_mean"] > 0:
                    pct = (row["recall@1_mean"] - ref.loc[N, "recall@1_mean"]
                           ) / ref.loc[N, "recall@1_mean"] * 100.0
                    pct_changes.append(pct)
    pct_abs_max = max(abs(p) for p in pct_changes) * 1.2
    ylim_pct = (-pct_abs_max, pct_abs_max)

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        height_ratios=[1, 1, 0.75],
        hspace=0.62, wspace=0.38,
        left=0.08, right=0.995,
        top=0.90,   bottom=0.07,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_trajectory(ax_a, df_A, "mimic", "recall@1", "Recall@1",
                    FULL_N["mimic"], show_ylabel=True,
                    shared_ylim=ylim_r1, letter="a")
    draw_trajectory(ax_b, df_A, "mimic", "mrr", "MRR",
                    FULL_N["mimic"], show_ylabel=False,
                    shared_ylim=ylim_mrr, letter="b")
    draw_pct_change_lines(ax_c, df_A, "mimic", FULL_N["mimic"],
                          show_ylabel=True, shared_ylim=ylim_pct, letter="c")

    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    draw_trajectory(ax_d, df_A, "chexpert", "recall@1", "Recall@1",
                    FULL_N["chexpert"], show_ylabel=True,
                    shared_ylim=ylim_r1, letter="d")
    draw_trajectory(ax_e, df_A, "chexpert", "mrr", "MRR",
                    FULL_N["chexpert"], show_ylabel=False,
                    shared_ylim=ylim_mrr, letter="e")
    draw_pct_change_lines(ax_f, df_A, "chexpert", FULL_N["chexpert"],
                          show_ylabel=True, shared_ylim=ylim_pct, letter="f")

    ax_g = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])
    ax_i = fig.add_subplot(gs[2, 2])

    draw_hardneg_fold_bars(ax_g, df_B, df_A, "mimic",
                           show_ylabel=True, letter="g",
                           title="Fold above chance — MIMIC-CXR")
    draw_hardneg_fold_bars(ax_h, df_B, df_A, "chexpert",
                           show_ylabel=False, letter="h",
                           title="Fold above chance — CheXpert Plus")
    draw_hardneg_pct_change(ax_i, df_A, df_B, letter="i")

    for yi, ds in zip([0.755, 0.455], DATASETS):
        fig.text(0.005, yi, DATASET_LABELS[ds],
                 fontsize=11, fontweight="bold",
                 va="center", ha="left", rotation=90)

    draw_legend(fig)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")
    plt.close(fig)



