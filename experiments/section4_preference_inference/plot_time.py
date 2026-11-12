"""Generate the Section 4 computation-time figure (`fig:computation time`).

Reads ``results/section4/time_results.csv`` (per-instance wall-clock time
for SGVI vs SOR) and produces a 1-row figure with one subplot per
cognitive_biases level. Each subplot shows side-by-side bars for the two
methods, with SOR decomposed into:

  (a) inference (hit-and-run sampling under feasible constraints)
  (b) inconsistency identification (resolve + remove conflicting constraints)

Error bars show 95% confidence intervals of the mean wall-clock time across
the aggregated (dataset x value_function_type x random seed) groups.

Output:
  figures_generated/section4/computation_time_stacked_CI.{jpg,pdf,eps}
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


CSV_PATH = f"{CODE_ROOT}/results/section4/time_results.csv"
OUT_DIR  = f"{CODE_ROOT}/figures_generated/section4"
OUT_NAME = "computation_time_stacked_CI"

PALETTE = {
    "SGVI":      "#EF9A9A",
    "SOR_main":  "#7DAEE0",
    "SOR_incon": "#BBD8F0",
}

plt.rcParams.update({
    "font.size":      11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


def mean_ci(x, confidence=0.95):
    """Return (mean, mean-h, mean+h) where h is the t-based CI half-width."""
    a = np.asarray(x.dropna())
    n = len(a)
    mean = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - h, mean + h


def load_and_prepare():
    df = pd.read_csv(CSV_PATH)
    # When bias = 0 there is no inconsistency to resolve, so fold incon time
    # into the main SOR time and zero-out the incon component.
    mask = df["cognitive_biases"] == 0
    df.loc[mask, "SOR_time"] = df.loc[mask, "SOR_time"] - df.loc[mask, "incon_check_time"]
    df.loc[mask, "incon_check_time"] = 0

    df["SOR_incon_time"] = df["incon_check_time"]
    df["SOR_other_time"] = (df["SOR_time"] - df["incon_check_time"]).clip(lower=0)
    return df


def aggregate(df):
    """One row per (cognitive_biases, preferences_num): mean + CI for SGVI, SOR total + components."""
    rows = []
    for (bias, pref), sub in df.groupby(["cognitive_biases", "preferences_num"]):
        m_sgvi, lo_sgvi, hi_sgvi = mean_ci(sub["SGVI_time"])
        m_sor,  lo_sor,  hi_sor  = mean_ci(sub["SOR_time"])
        rows.append({
            "cognitive_biases": bias,
            "preferences_num":  pref,
            "SGVI_mean":        m_sgvi,
            "SGVI_low":         lo_sgvi,
            "SGVI_high":        hi_sgvi,
            "SOR_mean":         m_sor,
            "SOR_low":          lo_sor,
            "SOR_high":         hi_sor,
            "SOR_other_mean":   sub["SOR_other_time"].mean(),
            "SOR_incon_mean":   sub["SOR_incon_time"].mean(),
        })
    return pd.DataFrame(rows)


def plot(df_plot):
    unique_biases = sorted(df_plot["cognitive_biases"].unique())
    fig, axes = plt.subplots(1, len(unique_biases),
                             figsize=(len(unique_biases) * 3.2, 4), sharey=True)
    if len(unique_biases) == 1:
        axes = [axes]

    bar_width = 3.5
    bar_offset = 1.75

    for i, bias in enumerate(unique_biases):
        sub = df_plot[df_plot["cognitive_biases"] == bias]
        ax = axes[i]
        x_sgvi = sub["preferences_num"] - bar_offset
        x_sor  = sub["preferences_num"] + bar_offset

        # SGVI bar
        ax.bar(x_sgvi, sub["SGVI_mean"], width=bar_width, edgecolor="none",
               color=PALETTE["SGVI"], label="The proposed approach")
        ax.errorbar(x_sgvi, sub["SGVI_mean"],
                    yerr=[sub["SGVI_mean"] - sub["SGVI_low"],
                          sub["SGVI_high"] - sub["SGVI_mean"]],
                    fmt='none', ecolor='black', elinewidth=0.8,
                    capsize=2, capthick=0.8)

        # SOR stacked bar: main part + inconsistency check
        ax.bar(x_sor, sub["SOR_other_mean"], width=bar_width, edgecolor="none",
               color=PALETTE["SOR_main"], label="SOR (main)")
        ax.bar(x_sor, sub["SOR_incon_mean"], width=bar_width, edgecolor="none",
               bottom=sub["SOR_other_mean"], color=PALETTE["SOR_incon"],
               label="SOR (inconsistency check)")
        ax.errorbar(x_sor, sub["SOR_mean"],
                    yerr=[sub["SOR_mean"] - sub["SOR_low"],
                          sub["SOR_high"] - sub["SOR_mean"]],
                    fmt='none', ecolor='black', elinewidth=0.8,
                    capsize=2, capthick=0.8)

        ax.set_title(f"Biased pref. info. ratio = {bias}")
        ax.set_xlabel("No. of pref. info.")
        if i == 0:
            ax.set_ylabel("Computation Time (s)")
        ax.set_xticks([10, 20, 30, 40])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["SGVI"],      edgecolor='black'),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["SOR_main"],  edgecolor='black'),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["SOR_incon"], edgecolor='black'),
    ]
    legend_labels = [
        "The proposed approach",
        "SOR (sampling-based inference)",
        "SOR (inconsistency identification)",
    ]
    fig.legend(legend_handles, legend_labels,
               loc="upper right", bbox_to_anchor=(1.0, 1.08),
               ncol=3, frameon=True)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("jpg", "pdf", "eps"):
        fig.savefig(f"{OUT_DIR}/{OUT_NAME}.{ext}", dpi=500, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = load_and_prepare()
    df_plot = aggregate(df)
    plot(df_plot)
    print(f"[saved] {OUT_DIR}/{OUT_NAME}.{{jpg,pdf,eps}}")
