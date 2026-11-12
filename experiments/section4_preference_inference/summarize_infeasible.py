"""Summarize SOR infeasibility statistics for Table F.8 of the supplement.

Reads ``results/section4/time_results.csv`` (produced by ``run_timing.py``),
which records per-instance SOR behaviour via two columns:

  - ``incon_check``   : 1 if the preference system was infeasible and the
                        inconsistency-resolution module was invoked, else 0.
  - ``min_incon_num`` : minimum number of conflicting constraints removed
                        before the system became feasible (0 if feasible).

The script groups by (preferences_num, cognitive_biases) and reports, for
each (F2, F3) cell:

  - count     : number of infeasible instances in the cell
  - fraction  : proportion of infeasible instances
  - mean / std: mean / standard deviation of removed constraints among the
                infeasible instances (dashes if the cell has none)

Output:  ``results/section4/infeasible_summary.csv``
"""

import os

import pandas as pd


# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


PREFERENCES_NUMS = [10, 20, 30, 40]
COGNITIVE_BIASES = [0.0, 0.1, 0.2, 0.3]

TIME_CSV_PATH    = f"{CODE_ROOT}/results/section4/time_results.csv"
SUMMARY_CSV_PATH = f"{CODE_ROOT}/results/section4/infeasible_summary.csv"


def main():
    df = pd.read_csv(TIME_CSV_PATH)
    rows = []
    for pn in PREFERENCES_NUMS:
        for bias in COGNITIVE_BIASES:
            cell = df[(df["preferences_num"] == pn)
                      & (df["cognitive_biases"] == bias)]
            total = len(cell)
            infeasible = cell[cell["incon_check"] == 1]
            count = len(infeasible)
            if count == 0:
                mean_removed = "-"
                std_removed  = "-"
            else:
                mean_removed = round(infeasible["min_incon_num"].mean(), 2)
                # ddof=1 (sample std) matches the convention used in Tab. F.8.
                std_removed  = round(infeasible["min_incon_num"].std(ddof=1), 2)
            fraction = round(count / total * 100, 2) if total else 0.0
            rows.append({
                "preferences_num": pn,
                "cognitive_biases": bias,
                "count": count,
                "total": total,
                "fraction_pct": fraction,
                "mean_removed": mean_removed,
                "std_removed": std_removed,
            })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SUMMARY_CSV_PATH), exist_ok=True)
    out.to_csv(SUMMARY_CSV_PATH, index=False)
    print(out.to_string(index=False))
    print(f"\n[saved] {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
