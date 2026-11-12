"""Section 5 sensitivity-analysis figures (Appendix H and Appendix I).

Appendix H (Computation Budget sensitivity):
    plot_CB()  ->  figures_generated/section5/CB.{jpg,pdf,eps}

Appendix I (MCTS exploration parameter C_p sensitivity), 2 figures:
    plot_Cp()  ->  figures_generated/section5/Policies_nm_summary_Cp.{jpg,pdf,eps}
                   figures_generated/section5/Policies_round_summary_Cp.{jpg,pdf,eps}

Appendix H reads two dedicated CSVs (``results_CBtest_MCTS_{Ent,Var}.csv``).
Appendix I reads the main ``results.csv`` and relabels MCTS method names so
each (variant, C_p) pair appears as its own curve.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


RESULTS_DIR = os.path.join(CODE_ROOT, "results", "section5")
OUT_DIR     = os.path.join(CODE_ROOT, "figures_generated", "section5")

METRICS = ["fVAR", "fPWI", "fRAI"]
ROUNDS_LIST = [2, 4, 6, 8]


def save_fig(fig, name, dpi=500):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("jpg", "pdf", "eps"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Appendix H: CB sensitivity
# ============================================================

def plot_CB():
    def load_cb(csv_name):
        df = pd.read_csv(os.path.join(RESULTS_DIR, csv_name))
        df = df.groupby(["method", "CB"], as_index=False).mean()
        return df.rename(columns={"var": "fVAR"})

    df_var = load_cb("results_CBtest_MCTS_Var.csv")
    df_ent = load_cb("results_CBtest_MCTS_Ent.csv")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 5), sharex=True)
    for j, metric in enumerate(METRICS):
        axes[j].plot(df_var["CB"], df_var[metric], marker='o', label="MCTS_Var")
        axes[j].plot(df_ent["CB"], df_ent[metric], marker='s', color="orange", label="MCTS_Ent")
        axes[j].set_xlabel("Computation Budget")
        axes[j].set_ylabel(metric)
        axes[j].grid(True)
        axes[j].set_xticks([100, 200, 300, 400, 500])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "CB")


# ============================================================
# Appendix I: Cp sensitivity
# ============================================================

CP_METHODS = [
    "MCTS_Var(Cp=0.2)", "MCTS_Ent(Cp=0.2)",
    "MCTS_Var(Cp=0.5)", "MCTS_Ent(Cp=0.5)",
    r"MCTS_Var(Cp=$1/\sqrt{2}$)", r"MCTS_Ent(Cp=$1/\sqrt{2}$)",
    "MCTS_Var(Cp=1.0)", "MCTS_Ent(Cp=1.0)",
    "H_PWI_d1", "H_PWI_d2",
    "H_RAI_d1", "H_RAI_d2",
    "H_DVF", "H_RAND",
]
CP_COLORS = plt.cm.inferno(np.linspace(0.1, 0.7, len(CP_METHODS)))
CP_MARKERS = ['s', '^', 'v', '+', '*', 'd', '+', '*', 's', 'o', '^', 'v', '+', '*']
CP_LINESTYLES = ['-'] * 8 + ['dotted'] * 6


def load_cp_results():
    """Read results.csv and relabel MCTS variants by their C_p value."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))
    df = df[df["method"] != "MCTS_Entropy"]
    df["method"] = df["method"].replace({
        "DVF":       "H_DVF",
        "RAND":      "H_RAND",
        "E_PWI_d1":  "H_PWI_d1",
        "E_PWI_d2":  "H_PWI_d2",
        "E_RAI_d1":  "H_RAI_d1",
        "E_RAI_d2":  "H_RAI_d2",
        "MCTS_Var_Cp_0.7071067811865475":     r"MCTS_Var(Cp=$1/\sqrt{2}$)",
        "MCTS_Entropy_Cp_0.7071067811865475": r"MCTS_Ent(Cp=$1/\sqrt{2}$)",
        "MCTS_Var_Cp_0.2":     "MCTS_Var(Cp=0.2)",
        "MCTS_Entropy_Cp_0.2": "MCTS_Ent(Cp=0.2)",
        "MCTS_Var_Cp_0.5":     "MCTS_Var(Cp=0.5)",
        "MCTS_Entropy_Cp_0.5": "MCTS_Ent(Cp=0.5)",
        "MCTS_Var_Cp_1.0":     "MCTS_Var(Cp=1.0)",
        "MCTS_Entropy_Cp_1.0": "MCTS_Ent(Cp=1.0)",
    })
    df = df.rename(columns={"var": "fVAR"})
    return df[df["random_index"] < 20]


def _plot_cp_nm_summary(df):
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
        "xtick.labelsize": 14, "ytick.labelsize": 15, "legend.fontsize": 10,
    })
    df_avg = df.groupby(["n", "m", "method"])[METRICS].mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharey=False)

    for j, metric in enumerate(METRICS):
        # Row 1: m=3, vary n
        ax = axes[0, j]
        sub = df_avg[df_avg["m"] == 3]
        for k, method in enumerate(CP_METHODS):
            data = sub[sub["method"] == method]
            ax.plot(data["n"], data[metric],
                    color=CP_COLORS[k], marker=CP_MARKERS[k], linestyle=CP_LINESTYLES[k],
                    lw=2, label=method)
        ax.set_title("(m=3)")
        ax.set_xlabel("n")
        ax.set_xticks([6, 7, 8, 9, 10])
        ax.set_ylabel(metric)

        # Row 2: n=8, vary m
        ax = axes[1, j]
        sub = df_avg[df_avg["n"] == 8]
        for k, method in enumerate(CP_METHODS):
            data = sub[sub["method"] == method]
            ax.plot(data["m"], data[metric],
                    color=CP_COLORS[k], marker=CP_MARKERS[k], linestyle=CP_LINESTYLES[k],
                    lw=2, label=method)
        ax.set_title("(n=8)")
        ax.set_xlabel("m")
        ax.set_xticks([2, 3, 4, 5])
        ax.set_ylabel(metric)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=8)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    save_fig(fig, "Policies_nm_summary_Cp")


def _plot_cp_round_summary(df):
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10,
    })
    df_avg = df.groupby(["method", "rounds"])[METRICS].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for j, metric in enumerate(METRICS):
        for k, method in enumerate(CP_METHODS):
            data = df_avg[df_avg["method"] == method]
            axes[j].plot(data["rounds"], data[metric],
                         color=CP_COLORS[k], marker=CP_MARKERS[k], linestyle=CP_LINESTYLES[k],
                         lw=2, label=method)
        axes[j].set_xlabel("Rounds")
        axes[j].set_xticks(ROUNDS_LIST)
        axes[j].set_ylabel(metric)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=8)
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    save_fig(fig, "Policies_round_summary_Cp")


def plot_Cp():
    df = load_cp_results()
    _plot_cp_nm_summary(df)
    _plot_cp_round_summary(df)


if __name__ == "__main__":
    plot_CB()
    plot_Cp()
    print(f"[saved] Appendix H (CB) + Appendix I (Cp) figures under {OUT_DIR}")
