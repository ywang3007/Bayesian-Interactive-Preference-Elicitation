"""
Aggregate per-round decision timing files into a method x T summary table.

Scans ``<results_root>/timings/`` for files named
``{method}_n{n}_m{m}_T{T}_random{idx}_F3{F3}_timing.pt`` (each a list of
per-round wall-clock seconds), and reports mean / median / std of the
per-round decision time by method x T. Run after a serial timing batch so
the numbers reflect per-method cost without contention.

Usage:
    python aggregate_timing.py \
        [--results_root results_timing] \
        [--F3s 0.0] \
        [--Ts 2 4 6 8]
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----



FNAME_RE = re.compile(
    r"^(?P<method>.+)_n(?P<n>\d+)_m(?P<m>\d+)_T(?P<T>\d+)"
    r"_random(?P<idx>\d+)_F3(?P<F3>[\d.]+)_timing\.pt$"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_root",
        default=os.path.join(CODE_ROOT, "results", "section5"),
        help="Directory with a 'timings/' subfolder (default: results/section5)",
    )
    p.add_argument("--F3s", type=float, nargs="*", default=None,
                   help="If set, only aggregate these F3 values")
    p.add_argument("--Ts", type=int, nargs="*", default=None,
                   help="If set, only aggregate these T values")
    p.add_argument("--methods", nargs="*", default=None,
                   help="Optional method filter")
    p.add_argument("--csv_out", default=None,
                   help="Output CSV path; default: <results_root>/timing_summary.csv "
                        "(or timing_per_instance.csv with --per_instance)")
    p.add_argument("--latex_out", default=None,
                   help="Optional LaTeX table path (requires pandas style)")
    p.add_argument("--per_instance", action="store_true",
                   help="R3: aggregate across-instance stats (mean/std/min/max of "
                        "per-instance means) instead of pooling round x instance. "
                        "This is the convention used in the paper's decision-time table.")
    return p.parse_args()


def main():
    args = parse_args()
    timing_dir = os.path.join(args.results_root, "timings")
    if not os.path.isdir(timing_dir):
        raise SystemExit(f"timings dir not found: {timing_dir}")

    rows = []
    for fname in sorted(os.listdir(timing_dir)):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        method = m.group("method")
        T = int(m.group("T"))
        F3 = float(m.group("F3"))
        idx = int(m.group("idx"))
        if args.Ts is not None and T not in args.Ts:
            continue
        if args.F3s is not None and not any(abs(F3 - f) < 1e-6 for f in args.F3s):
            continue
        if args.methods is not None and method not in args.methods:
            continue
        times = torch.load(os.path.join(timing_dir, fname))
        for round_i, t in enumerate(times):
            rows.append({
                "method": method,
                "T": T,
                "F3": F3,
                "random_index": idx,
                "round": round_i,
                "time_s": float(t),
            })

    if not rows:
        raise SystemExit("No matching timing files found.")

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} per-round records "
          f"from {df['method'].nunique()} methods, "
          f"{df['random_index'].nunique()} instances, "
          f"T={sorted(df['T'].unique())}, "
          f"F3={sorted(df['F3'].unique())}")

    if args.per_instance:
        # Table convention (R3, used for the paper's decision-time table):
        # (1) average over rounds within each (method, T, instance) -> per-instance mean
        # (2) then aggregate mean/std/min/max across instances
        # std here reflects across-instance variability, not the pooled
        # round+instance variability that the default mode reports.
        per_inst = (
            df.groupby(["method", "T", "random_index"])["time_s"]
              .mean()
              .reset_index()
        )
        summary = (
            per_inst.groupby(["method", "T"])["time_s"]
                    .agg(["mean", "std", "min", "max", "count"])
                    .reset_index()
        )
        for c in ["mean", "std", "min", "max"]:
            summary[c] = summary[c].round(3)
        summary = summary.rename(columns={"count": "n_instances"})
        default_csv_name = "timing_per_instance.csv"
    else:
        # Default: pool over (round x instance). Keeps backward compatibility
        # with the original R2 table.
        summary = (
            df.groupby(["method", "T"])["time_s"]
              .agg(["mean", "median", "std", "min", "max", "count"])
              .reset_index()
        )
        for c in ["mean", "median", "std", "min", "max"]:
            summary[c] = summary[c].round(3)
        default_csv_name = "timing_summary.csv"

    out_csv = args.csv_out or os.path.join(args.results_root, default_csv_name)
    summary.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")

    # Pivots for easy viewing
    pivot_mean = summary.pivot(index="method", columns="T", values="mean")
    print("\n=== Mean decision time (seconds) ===")
    print(pivot_mean.to_string())
    if "median" in summary.columns:
        pivot_median = summary.pivot(index="method", columns="T", values="median")
        print("\n=== Median per-round decision time (seconds) ===")
        print(pivot_median.to_string())
    if "std" in summary.columns:
        pivot_std = summary.pivot(index="method", columns="T", values="std")
        print("\n=== Std (across instances if --per_instance) decision time (seconds) ===")
        print(pivot_std.to_string())

    # Optional: dump a LaTeX-ready version
    if args.latex_out:
        latex = pivot_mean.to_latex(
            float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "--",
            caption="Per-round decision time (seconds), averaged over instances. "
                    "Serial timing on a single workstation.",
            label="tab:decision_time",
        )
        with open(args.latex_out, "w") as fp:
            fp.write(latex)
        print(f"[saved] {args.latex_out}")


if __name__ == "__main__":
    main()
