"""Section 5 main-text figures from ``results/section5/results.csv``.

Produces 4 figures (all under ``figures_generated/section5/``):

  A. ``sum_with_CI``             -- 1 x 3 bars: avg metric per method w/ 95% CI
  B. ``Policies_nm_summary``     -- 2 x 3 lines: metric vs. n (m=3) and vs. m (n=8)
  C. ``Policies_round_summary``  -- 1 x 3 lines: metric vs. rounds, averaged over n,m
  D. ``Policies_round_by_size``  -- 8 x 3 lines: metric vs. rounds per (n, m) setting (`fig:Policies_rounds`, R3 revision = per-problem-size layout)
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


# ---------- paths ----------
RESULTS_CSV = os.path.join(CODE_ROOT, "results", "section5", "results.csv")
OUT_DIR     = os.path.join(CODE_ROOT, "figures_generated", "section5")

# ---------- display configuration ----------
METHODS = ['MCTS_Var', 'MCTS_Ent', 'H_PWI_d1', 'H_RAI_d1',
           'H_PWI_d2', 'H_RAI_d2', 'H_DVF', 'H_RAND']
METRICS = ["fVAR", "fPWI", "fRAI"]
ROUNDS_LIST = [2, 4, 6, 8]

COLORS = plt.cm.inferno(np.linspace(0.1, 0.7, len(METHODS)))
MARKERS = ['s', 'o', '^', 'v', '+', '*', 'd', 'd']
LINESTYLES = ['-', '-', 'dotted', 'dotted', 'dotted', 'dotted', 'dotted', 'dotted']

# 8 problem settings used in Section 5: vary n at m=3, then vary m at n=8.
PROBLEM_SIZES = [
    (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
    (8, 2),         (8, 4), (8, 5),
]


# ---------- data loading ----------
def load_results():
    """Read ``results.csv``, drop legacy MCTS rows, rename to paper method
    names, and restrict to ``random_index < 20``."""
    df = pd.read_csv(RESULTS_CSV)
    df = df[~df["method"].isin(["MCTS_Entropy", "MCTS_Var"])]
    df["method"] = df["method"].replace({
        "MCTS_Var_Cp_0.7071067811865475":     "MCTS_Var",
        "MCTS_Entropy_Cp_0.7071067811865475": "MCTS_Ent",
        "DVF":       "H_DVF",
        "RAND":      "H_RAND",
        "E_PWI_d1":  "H_PWI_d1",
        "E_PWI_d2":  "H_PWI_d2",
        "E_RAI_d1":  "H_RAI_d1",
        "E_RAI_d2":  "H_RAI_d2",
    })
    df = df.rename(columns={"var": "fVAR"})
    return df[df["random_index"] < 20]


def save_fig(fig, name, dpi=500):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("jpg", "pdf", "eps"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------- Figure A: average metrics bar chart with 95% CI ----------
def plot_sum_with_ci(df):
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
        "xtick.labelsize": 14, "ytick.labelsize": 15,
        "legend.fontsize": 15, "legend.title_fontsize": 15,
    })

    stats = df.groupby("method")[METRICS].agg(["mean", "std", "count"]).reset_index()
    stats.columns = ["method"] + [f"{m}_{s}" for m, s in stats.columns[1:]]
    for metric in METRICS:
        n = stats[f"{metric}_count"]
        se = stats[f"{metric}_std"] / n ** 0.5
        stats[f"{metric}_ci"] = st.t.ppf(0.975, df=n - 1) * se

    stats = stats[stats["method"].isin(METHODS)].copy()
    stats["method"] = pd.Categorical(stats["method"], categories=METHODS, ordered=True)
    stats = stats.sort_values("method").reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for i, metric in enumerate(METRICS):
        axes[i].bar(stats["method"], stats[f"{metric}_mean"],
                    yerr=stats[f"{metric}_ci"],
                    color="skyblue", edgecolor="black", width=0.6,
                    error_kw={"elinewidth": 1, "capsize": 3, "capthick": 1})
        axes[i].set_title(metric)
        axes[i].set_xlabel("Questioning Policy")
        axes[i].set_ylabel("Average Value")
        axes[i].tick_params(axis="x", rotation=45)
        ymin = (stats[f"{metric}_mean"] - stats[f"{metric}_ci"]).min()
        ymax = (stats[f"{metric}_mean"] + stats[f"{metric}_ci"]).max()
        axes[i].set_ylim(ymin * 0.95, ymax * 1.05)

    plt.tight_layout()
    save_fig(fig, "sum_with_CI")


# ---------- Figure B: metric vs. n (m=3) and vs. m (n=8) ----------
def plot_nm_summary(df):
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
        "xtick.labelsize": 14, "ytick.labelsize": 15, "legend.fontsize": 14,
    })

    df_avg = df.groupby(["n", "m", "method"])[METRICS].mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharey=False)

    for j, metric in enumerate(METRICS):
        # Row 1: m=3, vary n
        ax = axes[0, j]
        sub = df_avg[df_avg["m"] == 3]
        for k, method in enumerate(METHODS):
            data = sub[sub["method"] == method]
            ax.plot(data["n"], data[metric],
                    color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLES[k],
                    lw=2, label=method)
        ax.set_title("(m=3)")
        ax.set_xlabel("n")
        ax.set_xticks([6, 7, 8, 9, 10])
        ax.set_ylabel(metric)

        # Row 2: n=8, vary m
        ax = axes[1, j]
        sub = df_avg[df_avg["n"] == 8]
        for k, method in enumerate(METHODS):
            data = sub[sub["method"] == method]
            ax.plot(data["m"], data[metric],
                    color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLES[k],
                    lw=2, label=method)
        ax.set_title("(n=8)")
        ax.set_xlabel("m")
        ax.set_xticks([2, 3, 4, 5])
        ax.set_ylabel(metric)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=8)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "Policies_nm_summary")


# ---------- Figure C: metric vs. rounds, averaged over (n, m) ----------
def plot_round_summary(df):
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })

    df_avg = df.groupby(["method", "rounds"])[METRICS].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

    for j, metric in enumerate(METRICS):
        for k, method in enumerate(METHODS):
            data = df_avg[df_avg["method"] == method]
            axes[j].plot(data["rounds"], data[metric],
                         color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLES[k],
                         lw=2, label=method)
        axes[j].set_xlabel("Rounds")
        axes[j].set_xticks(ROUNDS_LIST)
        axes[j].set_ylabel(metric)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=8)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    save_fig(fig, "Policies_round_summary")


# ---------- Figure D: metric vs. rounds, one row per (n, m) ----------
def plot_round_by_size(df):
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })

    df_per_size = (
        df.groupby(["n", "m", "method", "rounds"])[METRICS].mean().reset_index()
    )
    fig, axes = plt.subplots(
        len(PROBLEM_SIZES), len(METRICS),
        figsize=(5 * len(METRICS), 2.5 * len(PROBLEM_SIZES)),
        sharex=True, squeeze=False,
    )

    for i, (n, m) in enumerate(PROBLEM_SIZES):
        sub_size = df_per_size[(df_per_size["n"] == n) & (df_per_size["m"] == m)]
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for k, method in enumerate(METHODS):
                data = sub_size[sub_size["method"] == method].sort_values("rounds")
                if data.empty:
                    continue
                ax.plot(data["rounds"], data[metric],
                        color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLES[k],
                        lw=1.8, label=method)
            if i == 0:
                ax.set_title(metric)
            if j == 0:
                ax.set_ylabel(f"n={n}, m={m}", fontsize=12)
            if i == len(PROBLEM_SIZES) - 1:
                ax.set_xlabel("Rounds")
            ax.set_xticks(ROUNDS_LIST)
            ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(METHODS), bbox_to_anchor=(0.5, 1.00))
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save_fig(fig, "Policies_round_by_size", dpi=400)


if __name__ == "__main__":
    df = load_results()
    plot_sum_with_ci(df)
    plot_nm_summary(df)
    plot_round_summary(df)
    plot_round_by_size(df)
    print(f"[saved] 4 figures under {OUT_DIR}")
