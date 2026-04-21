"""
Created on April 20, 2026.
figure2.py

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



# Shared style
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


# Shared constants
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



# Shared utilities
def _parse_cell(cell):
    """Parse 'mean ± std [lo, hi]' → (mean, std, lo, hi)."""
    cell = str(cell).strip()
    m = re.match(
        r"([\d.]+)\s*±\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", cell
    )
    if m:
        return tuple(float(x) for x in m.groups())
    return (float(cell), 0.0, float(cell), float(cell))


def _load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["model_name"]   = df["model_name"].str.strip().replace({"Clip": "CLIP"})
    df["dataset_name"] = df["dataset_name"].str.strip()
    df["variant_name"] = df["variant_name"].str.strip()

    for col in ["recall@1", "recall@5", "recall@10", "mrr"]:
        parsed = df[col].apply(_parse_cell)
        df[f"{col}_mean"] = parsed.apply(lambda x: x[0])
        df[f"{col}_std"]  = parsed.apply(lambda x: x[1])
        df[f"{col}_lo"]   = parsed.apply(lambda x: x[2])
        df[f"{col}_hi"]   = parsed.apply(lambda x: x[3])
    return df


def _chance_recall1(N):
    return 100.0 / N


def _chance_mrr(N):
    """E[1/rank] for uniform random rank in 1..N, expressed as percent."""
    if N <= 5000:
        return 100.0 * np.sum(1.0 / np.arange(1, N + 1)) / N
    return 100.0 * (np.log(N) + 0.5772) / N


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _x_tick_labels(pools, full_N):
    labels = []
    for p in pools:
        if p == full_N:
            labels.append(f"{p:,}\n(full)")
        elif p >= 1000:
            labels.append(f"{p // 1000:,}K")
        else:
            labels.append(str(p))
    return labels





class Figure2:
    DATASETS     = ["mimic", "chexpert"]
    DATASET_LABELS = {"mimic": "MIMIC-CXR", "chexpert": "CheXpert Plus"}
    PANEL_LABELS = [["a", "b", "c"], ["d", "e", "f"]]

    def __init__(self, csv_path: str, out_path: str = "figure2.png"):
        self.csv_path = csv_path
        self.out_path = out_path
        self._df = None


    def plot(self):
        self._df = _load_data(self.csv_path)
        df_A = self._df[
            (self._df["variant_name"] == "off_shelf") &
            (self._df["setting"] == "A")
        ].copy()

        fig = plt.figure(figsize=(14, 8.5))
        gs  = gridspec.GridSpec(
            2, 3, figure=fig,
            hspace=0.58, wspace=0.38,
            left=0.08, right=0.995,
            top=0.88,   bottom=0.10,
        )

        for ri, dataset in enumerate(self.DATASETS):
            sub    = df_A[df_A["dataset_name"] == dataset].copy()
            full_N = int(sub["N"].max())
            labels = self.PANEL_LABELS[ri]

            ax_hier  = fig.add_subplot(gs[ri, 0])
            ax_fold  = fig.add_subplot(gs[ri, 1])
            ax_decay = fig.add_subplot(gs[ri, 2])

            self._draw_hierarchy(ax_hier,  sub[sub["N"] == 100], labels[0])
            self._draw_fold(ax_fold, sub, labels[1], full_N,
                            show_ylabel=True)
            self._draw_decay(ax_decay, sub, labels[2], full_N,
                             show_ylabel=True)

        # Row labels
        for yi, dataset in zip([0.73, 0.28], self.DATASETS):
            fig.text(0.005, yi, self.DATASET_LABELS[dataset],
                     fontsize=11, fontweight="bold",
                     va="center", ha="left", rotation=90)

        self._draw_legend(fig)

        fig.savefig(self.out_path, dpi=300,
                    bbox_inches="tight", facecolor="white")
        print(f"Saved {self.out_path}")
        plt.close(fig)


    # Panel helpers
    def _draw_hierarchy(self, ax, sub, label):
        bar_h = 0.32
        gap   = 0.08
        pos_r1  = np.arange(len(MODEL_ORDER)) * 1.0
        pos_mrr = pos_r1 + bar_h + gap

        for i, model in enumerate(MODEL_ORDER):
            row = sub[sub["model_name"] == model]
            if row.empty:
                continue
            r1_m  = row["recall@1_mean"].values[0]
            r1_s  = row["recall@1_std"].values[0]
            mrr_m = row["mrr_mean"].values[0]
            mrr_s = row["mrr_std"].values[0]
            c     = MODEL_COLORS[model]

            ax.barh(pos_r1[i],  r1_m,  height=bar_h,
                    color=c, alpha=0.90, zorder=3)
            ax.errorbar(r1_m,  pos_r1[i],  xerr=r1_s,
                        fmt="none", color="black",
                        elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)

            ax.barh(pos_mrr[i], mrr_m, height=bar_h,
                    color=c, alpha=0.45, hatch="///",
                    edgecolor=c, linewidth=0.5, zorder=3)
            ax.errorbar(mrr_m, pos_mrr[i], xerr=mrr_s,
                        fmt="none", color="black",
                        elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)

        ch_r1  = _chance_recall1(100)
        ch_mrr = _chance_mrr(100)
        ax.axvline(ch_r1,  color="#555555", lw=0.9, ls="--", zorder=2)
        ax.axvline(ch_mrr, color="#555555", lw=0.9, ls=":",  zorder=2)

        n = len(MODEL_ORDER)
        ax.text(ch_r1  + 0.3, n - 0.1, "chance\nR@1",
                fontsize=6.5, color="#555555", va="top", ha="left")
        ax.text(ch_mrr + 0.3, n - 0.1, "chance\nMRR",
                fontsize=6.5, color="#555555", va="top", ha="left")

        ytick_pos = (pos_r1 + pos_mrr) / 2
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(MODEL_ORDER, fontsize=9)
        ax.set_ylim(-0.3, n - 0.1)
        ax.set_xlabel("Performance (%)", fontsize=10, labelpad=3)
        ax.text(-0.11, 1.06, "Linkage hierarchy ($N = 100$)",
                transform=ax.transAxes, fontsize=10, va="bottom", ha="left")
        ax.text(-0.22, 1.06, f"$\\mathbf{{{label}}}$",
                transform=ax.transAxes, fontsize=16, va="bottom", ha="left")
        _despine(ax)

    def _draw_fold(self, ax, sub, label, full_N, show_ylabel):
        pools = sorted(sub["N"].unique())

        for model in MODEL_ORDER:
            sub_m = sub[sub["model_name"] == model].sort_values("N")
            if sub_m.empty:
                continue
            xs, ys, stds = [], [], []
            for _, row in sub_m.iterrows():
                ch = _chance_recall1(row["N"])
                fold = row["recall@1_mean"] / ch
                fold_std = row["recall@1_std"] / ch  # Scale std by same factor
                xs.append(row["N"])
                ys.append(fold)
                stds.append(fold_std)

            xs = np.array(xs)
            ys = np.array(ys)
            stds = np.array(stds)

            ax.errorbar(
                xs, ys, yerr=stds,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markersize=MODEL_MS[model],
                linewidth=1.4, capsize=2.2,
                capthick=0.9, elinewidth=0.7,
                zorder=3 + MODEL_ORDER.index(model),
                label=model,
            )

        ax.axhline(1.0, color="#888888", lw=0.9, ls="--", zorder=2)
        ax.text(pools[0] * 1.15, 1.08, "1× chance",
                fontsize=7.5, color="#888888", va="bottom")
        ax.axhspan(0, 1.0, color="#cccccc", alpha=0.15, zorder=0)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.2f}"
        ))
        ax.set_xticks(pools)
        ax.set_xticklabels(_x_tick_labels(pools, full_N), fontsize=8)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel("Candidate-pool size $N$", fontsize=10, labelpad=3)
        if show_ylabel:
            ax.set_ylabel("Recall@1 / chance Recall@1",
                          fontsize=9.5, labelpad=3)
        ax.text(0.02, 1.06, "Fold above chance",
                transform=ax.transAxes, fontsize=10, va="bottom", ha="left")
        ax.text(-0.14, 1.06,
                f"$\\mathbf{{{label}}}$",
                transform=ax.transAxes, fontsize=16, va="bottom", ha="left")
        _despine(ax)


    def _draw_decay(self, ax, sub, label, full_N, show_ylabel):
        pools = sorted(sub["N"].unique())

        for model in MODEL_ORDER:
            sub_m = sub[sub["model_name"] == model].sort_values("N")
            if sub_m.empty:
                continue
            xs    = sub_m["N"].values
            c     = MODEL_COLORS[model]
            mk    = MODEL_MARKERS[model]
            ms    = MODEL_MS[model]

            ax.errorbar(xs, sub_m["recall@1_mean"].values,
                        yerr=sub_m["recall@1_std"].values,
                        color=c, marker=mk, markersize=ms,
                        linewidth=1.4, linestyle="-",
                        capsize=2.2, capthick=0.9, elinewidth=0.7,
                        zorder=3 + MODEL_ORDER.index(model))
            ax.errorbar(xs, sub_m["mrr_mean"].values,
                        yerr=sub_m["mrr_std"].values,
                        color=c, marker=mk, markersize=ms * 0.75,
                        linewidth=1.0, linestyle="--", alpha=0.65,
                        capsize=1.8, capthick=0.7, elinewidth=0.6,
                        zorder=3 + MODEL_ORDER.index(model))

        ch_r1  = [_chance_recall1(N) for N in pools]
        ch_mrr = [_chance_mrr(N)     for N in pools]
        ax.plot(pools, ch_r1,  color="#888888", lw=0.9, ls="-",  zorder=2)
        ax.plot(pools, ch_mrr, color="#888888", lw=0.9, ls="--", zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xticks(pools)
        ax.set_xticklabels(_x_tick_labels(pools, full_N), fontsize=8)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.set_xlabel("Candidate-pool size $N$", fontsize=10, labelpad=3)


        if show_ylabel:
            ax.set_ylabel("Performance (%)", fontsize=10, labelpad=3)
        ax.text(-0.02, 1.06, "Retrieval decay (solid = Recall@1, dashed = MRR)",
                transform=ax.transAxes, fontsize=10, va="bottom", ha="left")
        ax.text(-0.14, 1.06,
                f"$\\mathbf{{{label}}}$",
                transform=ax.transAxes, fontsize=16, va="bottom", ha="left")
        _despine(ax)

    def _draw_legend(self, fig):
        model_handles = [
            Line2D([0], [0], color=MODEL_COLORS[m],
                   marker=MODEL_MARKERS[m], markersize=MODEL_MS[m],
                   linewidth=1.4, label=m)
            for m in MODEL_ORDER
        ]
        metric_handles = [
            Line2D([0], [0], color="black", lw=1.4, ls="-",
                   label="Recall@1"),
            Line2D([0], [0], color="black", lw=1.0, ls="--",
                   label="MRR (panels c, f)"),
            Patch(facecolor="#555555", alpha=0.7,
                  label="Recall@1 bar (panels a, d)"),
            Patch(facecolor="#888888", alpha=0.35, hatch="///",
                  edgecolor="#555555",
                  label="MRR bar (panels a, d)"),
        ]
        fig.legend(
            handles=model_handles + metric_handles,
            loc="upper center", ncol=8,
            fontsize=10, frameon=False,
            bbox_to_anchor=(0.53, 0.975),
            handlelength=1.8, handletextpad=0.45, columnspacing=1.0,
        )


