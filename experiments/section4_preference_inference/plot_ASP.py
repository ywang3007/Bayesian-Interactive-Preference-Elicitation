"""Generate Section 4 ASP figures (Figures 4, 5, 6).

Reads ``results/section4/ASP_results.csv`` (per-instance ASP for SGVI vs SOR)
and produces:

  - F1 (`fig:ASPresults_F1`): boxplot by value-function type, 8 datasets subplots
       → figures_generated/section4/dataset_F1.{jpg,pdf,eps}
  - F2 (`fig:ASPresults_F2`): line plot by preferences_num,     8 datasets subplots
       → figures_generated/section4/dataset_F2.{jpg,pdf,eps}
  - F3 (`fig:ASPresults_F3`): line plot by cognitive_biases,    8 datasets subplots
       → figures_generated/section4/dataset_F3.{jpg,pdf,eps}

The CSV is filtered to preferences_num in {10, 20, 30, 40} (the four values
reported in the paper) before plotting.
"""

import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


# ---------- display configuration ----------
DATASET_NAMES = ['TC', 'IC', 'SS', 'CE', 'RU', 'EBC', 'MPI', 'GUU']
VALUE_FUNCTION_TYPES = ['linear', 'concave', 'convex', 'concave&convex']
PREFERENCES_NUMS_PLOT = [10, 20, 30, 40]
COGNITIVE_BIASES_PLOT = [0.0, 0.1, 0.2, 0.3]

COLORS_BOXPLOT = {
    'SGVI_edge':  '#020101',
    'SGVI_face':  '#FFEBEE',
    'SOR_edge':   '#0D47A1',
    'SOR_face':   '#E3F2FD',
}
COLORS_LINE = {
    'SGVI': '#EF9A9A',
    'SOR':  '#7DAEE0',
}

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
})

FIG_SIZE = (16, 7)
OUT_DIR = f"{CODE_ROOT}/figures_generated/section4"


# ---------- data loading ----------
def load_asp_results():
    df = pd.read_csv(f"{CODE_ROOT}/results/section4/ASP_results.csv")
    return df[df["preferences_num"].isin(PREFERENCES_NUMS_PLOT)]


def save_fig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("jpg", "pdf", "eps"):
        fig.savefig(f"{OUT_DIR}/{name}.{ext}", dpi=500, bbox_inches="tight")
    plt.close(fig)


# ---------- F1: boxplot by value-function type ----------
def plot_F1(df):
    x = np.arange(len(VALUE_FUNCTION_TYPES))
    fig, ax = plt.subplots(2, 4, figsize=FIG_SIZE)
    for data_set_index, name in enumerate(DATASET_NAMES):
        i, j = divmod(data_set_index, 4)
        sub = df[df["data_set_index"] == data_set_index]

        y_sgvi = [sub[sub["value_function_type"] == vft]["ASP_SGVI_repara"].values
                  for vft in VALUE_FUNCTION_TYPES]
        y_sor  = [sub[sub["value_function_type"] == vft]["ASP_SOR"].values
                  for vft in VALUE_FUNCTION_TYPES]

        bp_sgvi = ax[i, j].boxplot(
            [np.asarray(v).flatten() for v in y_sgvi],
            positions=x - 0.20, widths=0.16, showmeans=True, meanline=True,
            patch_artist=True, showfliers=False,
            boxprops={'color': COLORS_BOXPLOT['SGVI_edge'],
                      'facecolor': COLORS_BOXPLOT['SGVI_face']},
            medianprops={'linestyle': '-', 'color': 'black'},
            meanprops={'linestyle': '--', 'color': 'black'},
        )
        bp_sor = ax[i, j].boxplot(
            [np.asarray(v).flatten() for v in y_sor],
            positions=x + 0.04, widths=0.16, showmeans=True, meanline=True,
            patch_artist=True, showfliers=False,
            boxprops={'color': COLORS_BOXPLOT['SOR_edge'],
                      'facecolor': COLORS_BOXPLOT['SOR_face']},
            medianprops={'linestyle': '-', 'color': 'black'},
            meanprops={'linestyle': '--', 'color': 'black'},
        )

        ax[i, j].set_xticks(x)
        ax[i, j].set_xticklabels(VALUE_FUNCTION_TYPES[:3] + ['mixture'])
        ax[i, j].set_xlabel('Shape of function')
        ax[i, j].set_ylabel('ASP')
        ax[i, j].set_title(f'Experiment on dataset {name}')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.legend(
        [bp_sgvi["boxes"][0], bp_sor["boxes"][0]],
        ['The proposed approach', 'SOR'],
        loc='upper right', ncol=3,
    )
    save_fig(fig, 'dataset_F1')


# ---------- F2 / F3: line plot with 95% CI error bars ----------
def _mean_and_ci95(values):
    """95% CI half-width assuming the sample mean is t-distributed."""
    v = np.asarray(values)
    return 1.96 * np.std(v, ddof=1) / math.sqrt(len(v))


def _plot_lines_per_dataset(df, factor_col, factor_values, xlabel, out_name,
                             show_error_bars=False):
    x = np.arange(len(factor_values))
    fig, ax = plt.subplots(2, 4, figsize=FIG_SIZE)
    for data_set_index, name in enumerate(DATASET_NAMES):
        i, j = divmod(data_set_index, 4)
        sub = df[df["data_set_index"] == data_set_index]

        y_sgvi, y_sor, err_sgvi, err_sor = [], [], [], []
        for v in factor_values:
            g_sgvi = sub[sub[factor_col] == v]["ASP_SGVI_repara"].values
            g_sor  = sub[sub[factor_col] == v]["ASP_SOR"].values
            y_sgvi.append(g_sgvi.mean())
            y_sor.append(g_sor.mean())
            err_sgvi.append(_mean_and_ci95(g_sgvi))
            err_sor.append(_mean_and_ci95(g_sor))

        ax[i, j].plot(x, y_sgvi, color=COLORS_LINE['SGVI'], marker='s', lw=2,
                      label="The proposed approach")
        ax[i, j].plot(x, y_sor,  color=COLORS_LINE['SOR'],  marker='*', lw=2,
                      label="SOR")
        if show_error_bars:
            ax[i, j].errorbar(x, y_sgvi, yerr=err_sgvi, fmt='none', ecolor='black',
                              elinewidth=0.8, capsize=2)
            ax[i, j].errorbar(x, y_sor,  yerr=err_sor,  fmt='none', ecolor='black',
                              elinewidth=0.8, capsize=2)

        ax[i, j].set_xticks(x)
        ax[i, j].set_xticklabels(factor_values)
        ax[i, j].set_xlabel(xlabel)
        ax[i, j].set_ylabel('ASP')
        ax[i, j].set_title(f'Experiment on dataset {name}')

    handle, label = ax[0, 0].get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.legend(handle, label, loc='upper right', ncol=3)
    save_fig(fig, out_name)


def plot_F2(df):
    _plot_lines_per_dataset(df, "preferences_num", PREFERENCES_NUMS_PLOT,
                            xlabel="No. of pref. info.", out_name="dataset_F2")


def plot_F3(df):
    _plot_lines_per_dataset(df, "cognitive_biases", COGNITIVE_BIASES_PLOT,
                            xlabel="Biased pref. info. ratio", out_name="dataset_F3")


if __name__ == "__main__":
    df = load_asp_results()
    plot_F1(df)
    plot_F2(df)
    plot_F3(df)
    print(f"[saved] F1 / F2 / F3 figures under {OUT_DIR}")
