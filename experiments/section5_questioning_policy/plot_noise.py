"""
Generate the Appendix J figure for the noise-robustness experiment.

Layout: 3 rows (fVAR, fPWI, fRAI) x 4 columns (F3 = 0, 0.1, 0.2, 0.3).
Each panel: x = interaction horizon T, one line per method (8 total),
error bar = SEM across 20 instances at (n=8, m=3).

F3 = 0 is taken from the main-paper baseline CSV
(``results/section5/results.csv``).
F3 in {0.1, 0.2, 0.3} is taken from the noise experiment CSV
(``results/section5/eval_results_noise.csv``).

Output: ``figures_generated/section5/noise/appendix_noise_by_F3.{png,pdf}``.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----



# Mirrors strategies_202509/plot.py
BASELINE_METHOD_RENAME = {
    "MCTS_Var_Cp_0.7071067811865475": "MCTS_Var",
    "MCTS_Entropy_Cp_0.7071067811865475": "MCTS_Ent",
    "E_PWI_d1": "H_PWI_d1",
    "E_PWI_d2": "H_PWI_d2",
    "E_RAI_d1": "H_RAI_d1",
    "E_RAI_d2": "H_RAI_d2",
    "DVF": "H_DVF",
    "RAND": "H_RAND",
}
BASELINE_METHODS_TO_DROP = {"MCTS_Entropy", "MCTS_Var"}

METHOD_ORDER = [
    "MCTS_Var", "MCTS_Ent",
    "H_PWI_d1", "H_RAI_d1",
    "H_PWI_d2", "H_RAI_d2",
    "H_DVF", "H_RAND",
]

# Metric key in CSV -> display label
METRICS = [("var", "fVAR"), ("fPWI", "fPWI"), ("fRAI", "fRAI")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--noise_csv",
        default=os.path.join(CODE_ROOT, "results", "section5",
                             "eval_results_noise.csv"),
    )
    p.add_argument(
        "--baseline_csv",
        default=os.path.join(CODE_ROOT, "results", "section5", "results.csv"),
    )
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--Ts", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--F3s", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3])
    p.add_argument("--out_png", default=None)
    return p.parse_args()


def load_baseline(path, n, m, Ts):
    if not os.path.exists(path):
        print(f"[warn] baseline csv not found: {path}; F3=0 will be empty")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[~df["method"].isin(BASELINE_METHODS_TO_DROP)].copy()
    df["method"] = df["method"].map(BASELINE_METHOD_RENAME).fillna(df["method"])
    df = df[(df["n"] == n) & (df["m"] == m) & df["rounds"].isin(Ts)].copy()
    df = df.rename(columns={"rounds": "T"})
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df = df[df["random_index"] < 20].copy()
    df["F3"] = 0.0
    keep = ["n", "m", "method", "T", "random_index", "F3",
            "var", "fPWI", "fRAI"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def load_noise(path, n, m, Ts, F3s_wanted):
    if not os.path.exists(path):
        sys.exit(f"noise csv not found: {path}")
    df = pd.read_csv(path)
    df["F3"] = df["F3"].astype(float).round(2)
    df["T"] = df["T"].astype(int)
    pos_F3 = [round(f, 2) for f in F3s_wanted if f > 0]
    df = df[
        (df["n"] == n) & (df["m"] == m)
        & df["T"].isin(Ts)
        & df["method"].isin(METHOD_ORDER)
        & df["F3"].isin(pos_F3)
    ].copy()
    keep = ["n", "m", "method", "T", "random_index", "F3",
            "var", "fPWI", "fRAI"]
    return df[keep]


def main():
    args = parse_args()

    df_base = load_baseline(args.baseline_csv, args.n, args.m, args.Ts)
    df_noise = load_noise(args.noise_csv, args.n, args.m, args.Ts, args.F3s)
    df = pd.concat([df_base, df_noise], ignore_index=True)
    if df.empty:
        sys.exit("No rows available after filtering.")

    print(
        f"Loaded {len(df)} rows "
        f"(baseline F3=0: {len(df_base)}, noise F3>0: {len(df_noise)}) | "
        f"methods={sorted(df['method'].unique())} | "
        f"T={sorted(df['T'].unique())} | "
        f"F3={sorted(df['F3'].unique())}"
    )

    # ---- styling (matches strategies_202509/plot.py) ----
    colors_all = plt.cm.inferno(np.linspace(0.1, 0.7, len(METHOD_ORDER)))
    markers_all = ["s", "o", "^", "v", "+", "*", "d", "D"]
    linestyles_all = ["-", "-", "dotted", "dotted",
                      "dotted", "dotted", "dotted", "dotted"]
    style = {m: (colors_all[i], markers_all[i], linestyles_all[i])
             for i, m in enumerate(METHOD_ORDER)}

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    F3_levels = sorted(df["F3"].unique())
    fig, axes = plt.subplots(
        len(METRICS), len(F3_levels),
        figsize=(4.0 * len(F3_levels), 3.2 * len(METRICS)),
        squeeze=False,
    )

    for row_i, (metric_key, metric_label) in enumerate(METRICS):
        for col_i, F3 in enumerate(F3_levels):
            ax = axes[row_i, col_i]
            sub = df[df["F3"] == F3]
            if metric_key not in sub.columns:
                ax.set_visible(False)
                continue
            for method in METHOD_ORDER:
                m_sub = sub[sub["method"] == method]
                if m_sub.empty or m_sub[metric_key].isna().all():
                    continue
                agg = (
                    m_sub.groupby("T")[metric_key]
                         .mean()
                         .reset_index()
                         .sort_values("T")
                )
                x = agg["T"].values
                y = agg[metric_key].values
                c, mk, ls = style.get(method, (None, "o", "-"))
                ax.plot(
                    x, y, marker=mk, linestyle=ls,
                    color=c, lw=1.6, markersize=6, label=method,
                )
            if row_i == 0:
                ax.set_title(f"Biased pref. info. ratio = {F3:.2f}")
            if col_i == 0:
                ax.set_ylabel(metric_label)
            if row_i == len(METRICS) - 1:
                ax.set_xlabel("Rounds $T$")
            ax.set_xticks(args.Ts)
            ax.grid(alpha=0.3)

    # Shared legend from the first non-empty panel
    handles, labels = [], []
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(METHOD_ORDER), bbox_to_anchor=(0.5, 1.00))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = args.out_png or os.path.join(
        CODE_ROOT, "figures_generated", "section5", "noise",
        "appendix_noise_by_F3.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    out_pdf = out_png.replace(".png", ".pdf")
    out_eps = out_png.replace(".png", ".eps")
    plt.savefig(out_pdf, bbox_inches="tight")  # vector for LaTeX
    plt.savefig(out_eps, bbox_inches="tight")  # vector for journal submission
    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_eps}")


if __name__ == "__main__":
    main()
