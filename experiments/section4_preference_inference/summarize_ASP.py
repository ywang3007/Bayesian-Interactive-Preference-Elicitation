"""Summarize per-instance ASP into the Section 4 summary table + Wilcoxon tables.

Reads the per-instance CSV produced by ``compute_ASP.py``:

    results/section4/ASP_results.csv

Filters to ``preferences_num in {10, 20, 30, 40}`` (the four settings
reported in the paper), then produces three CSVs:

1. ``results/section4/ASP_table.csv`` (feeds Tab. 4 of the main paper)
   Per-dataset mean ASP for SGVI and SOR, broken down by F1 / F2 / F3.

2. ``results/section4/wilcoxon_per_dataset.csv`` (feeds Tab. E.5)
   One row per dataset: statistic and p-value of the one-sided Wilcoxon
   signed-rank test (H1: SGVI > SOR), aggregated over all settings.

3. ``results/section4/wilcoxon_by_setting.csv`` (feeds Tab. E.6)
   One row per (setting_type, setting_value), with 8 p-value columns
   (one per dataset). Setting_type ∈ {F1, F2, F3}.
"""

import os

import pandas as pd
import scipy.stats as stats


# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


# ---------- summary configuration ----------
DATASETS = range(8)
DATASET_NAMES = ['TC', 'IC', 'SS', 'CE', 'RU', 'EBC', 'MPI', 'GUU']
VALUE_FUNCTION_TYPES = ['linear', 'concave', 'convex', 'concave&convex']
PREFERENCES_NUMS_TABLE = [10, 20, 30, 40]
COGNITIVE_BIASES = [0, 0.1, 0.2, 0.3]

# ---------- I/O paths ----------
ASP_CSV_PATH          = f"{CODE_ROOT}/results/section4/ASP_results.csv"
TABLE_CSV_PATH        = f"{CODE_ROOT}/results/section4/ASP_table.csv"
WILCOXON_DS_CSV_PATH  = f"{CODE_ROOT}/results/section4/wilcoxon_per_dataset.csv"
WILCOXON_SET_CSV_PATH = f"{CODE_ROOT}/results/section4/wilcoxon_by_setting.csv"


def _method_row(method_tag, sub_df, col):
    """Build one row of the summary table for a given method (RSGVI or SOR)."""
    row = {"method": method_tag}
    for vft in VALUE_FUNCTION_TYPES:
        row[f"F1_{vft}"] = sub_df[sub_df["value_function_type"] == vft][col].mean()
    for pn in PREFERENCES_NUMS_TABLE:
        row[f"F2_{pn}"] = sub_df[sub_df["preferences_num"] == pn][col].mean()
    for cb in COGNITIVE_BIASES:
        row[f"F3_{cb}"] = sub_df[sub_df["cognitive_biases"] == cb][col].mean()
    row["Avg"] = sub_df[col].mean()
    return row


def _wilcoxon_one_sided(sub):
    """Return (statistic, p-value) for H1: SGVI > SOR on this sub-frame."""
    return stats.wilcoxon(
        sub["ASP_SGVI_repara"].values,
        sub["ASP_SOR"].values,
        alternative="greater",
    )


if __name__ == "__main__":
    df = pd.read_csv(ASP_CSV_PATH)
    df = df[df["preferences_num"].isin(PREFERENCES_NUMS_TABLE)]

    # ---------- Tab. 4 (ASP summary) + Tab. E.5 (per-dataset Wilcoxon) ----------
    rows = []
    wilcoxon_ds_rows = []
    for data_set_index in DATASETS:
        sub = df[df["data_set_index"] == data_set_index]
        row_sgvi = _method_row("RSGVI", sub, "ASP_SGVI_repara")
        row_sor  = _method_row("SOR",   sub, "ASP_SOR")
        row_sgvi["data_set_index"] = data_set_index
        row_sor ["data_set_index"] = data_set_index
        rows.append(row_sgvi)
        rows.append(row_sor)

        stat, p = _wilcoxon_one_sided(sub)
        wilcoxon_ds_rows.append({
            "data_set_index": data_set_index,
            "dataset_name":   DATASET_NAMES[data_set_index],
            "statistic":      stat,
            "p_value":        p,
        })
        print(f"dataset{data_set_index} ({DATASET_NAMES[data_set_index]}): "
              f"Wilcoxon stat={stat:.3f}, p={p:.4e}")

    os.makedirs(os.path.dirname(TABLE_CSV_PATH), exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE_CSV_PATH, index=False)
    pd.DataFrame(wilcoxon_ds_rows).to_csv(WILCOXON_DS_CSV_PATH, index=False)
    print(f"[saved] {TABLE_CSV_PATH}  ({len(rows)} rows)")
    print(f"[saved] {WILCOXON_DS_CSV_PATH}  ({len(wilcoxon_ds_rows)} rows)")

    # ---------- Tab. E.6 (per-setting Wilcoxon, wide-form) ----------
    factor_specs = [
        ("F1", "value_function_type", VALUE_FUNCTION_TYPES),
        ("F2", "preferences_num",     PREFERENCES_NUMS_TABLE),
        ("F3", "cognitive_biases",    COGNITIVE_BIASES),
    ]
    wilcoxon_set_rows = []
    for setting_type, col, levels in factor_specs:
        for level in levels:
            row = {"setting_type": setting_type, "setting_value": level}
            for data_set_index in DATASETS:
                sub = df[(df["data_set_index"] == data_set_index)
                         & (df[col] == level)]
                if len(sub) == 0:
                    row[DATASET_NAMES[data_set_index]] = float("nan")
                    continue
                _, p = _wilcoxon_one_sided(sub)
                row[DATASET_NAMES[data_set_index]] = p
            wilcoxon_set_rows.append(row)
    pd.DataFrame(wilcoxon_set_rows).to_csv(WILCOXON_SET_CSV_PATH, index=False)
    print(f"[saved] {WILCOXON_SET_CSV_PATH}  ({len(wilcoxon_set_rows)} rows)")
