"""Measure per-instance computation time of the proposed SGVI method vs SOR
for preference inference (the `fig:computation time` figure (Section 4.3)).

  - 8 real-world datasets (mmc2--mmc9)
  - 4 value-function types
  - 4 preference-count levels: 10, 20, 30, 40
  - 4 bias ratios:              0.0, 0.1, 0.2, 0.3
  - 10 random seeds per combination
  -> 5,120 (SGVI, SOR) timing pairs

Each row records wall-clock time of one SGVI fit (``T_VI=100`` Adam iterations,
``W_g=10000`` MC samples) and the corresponding SOR hit-and-run sampling with
inconsistency detection. SOR time is decomposed into (a) inconsistency-check
time and (b) the rest, used for the stacked bars in the figure.
"""

import sys
import os
import json
import time
import random

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

sys.path.append(f"{CODE_ROOT}/core")
from SGVI import SGVI
from U_Sampler import u_sampler_SOR_for_time_test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------- experimental grid (matches the shipped CSV) ----------
DATASETS = range(8)
VALUE_FUNCTION_TYPES = ['linear', 'concave', 'convex', 'concave&convex']
PREFERENCES_NUMS = [10, 20, 30, 40]
COGNITIVE_BIASES = [0, 0.1, 0.2, 0.3]
N_INSTANCES = 10

# ---------- SGVI hyperparameters (match Section 4 in the paper) ----------
E = 1e-5         # convergence tolerance
T_VI = 100       # max Adam iterations
W_G = 10000      # MC samples for the stochastic gradient

# ---------- I/O paths ----------
SETTINGS_DIR = f"{CODE_ROOT}/data/section4_settings"
SGVI_OUT_DIR = f"{CODE_ROOT}/results/section4_sgvi_outputs"
CSV_PATH = f"{CODE_ROOT}/results/section4/time_results.csv"


def read_data_info(data_set_index, value_function_type, preferences_num, cognitive_biases, random_index):
    """Load one instance's eval matrix + observed preferences + pre-selected gamma."""
    settings_path = (f"{SETTINGS_DIR}/settings_dataset{data_set_index}"
                     f"_{value_function_type}_{preferences_num}_{cognitive_biases:f}"
                     f"_random{random_index}.json")
    with open(settings_path, 'r') as fp:
        setting = json.load(fp)
    evalmatrix = torch.tensor(setting['evalmatrix']).to(device)
    preferences_observed = torch.tensor(setting['preferences_observed']).to(device)

    sgvi_out_path = (f"{SGVI_OUT_DIR}/SGVIresults_dataset{data_set_index}"
                     f"_{value_function_type}_{preferences_num}_{cognitive_biases:f}"
                     f"_random{random_index}.pt")
    best_intervals_num = torch.load(sgvi_out_path)['best_intervals_num']
    return evalmatrix, preferences_observed, best_intervals_num


def load_done_set():
    """Resumability: keep every combination already present in the CSV so we can skip it."""
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        df = pd.DataFrame(columns=[
            "data_set_index", "value_function_type", "preferences_num",
            "cognitive_biases", "random_index", "SGVI_time", "SOR_time",
            "min_incon_num", "incon_check", "incon_check_time",
        ])
    else:
        df = pd.read_csv(CSV_PATH)
    done = set(zip(df['data_set_index'], df['value_function_type'],
                   df['preferences_num'], df['cognitive_biases'],
                   df['random_index']))
    return df, done


def flush(df_old, records):
    df_new = pd.concat([df_old, pd.DataFrame(records)], ignore_index=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_new.to_csv(CSV_PATH, index=False)
    return df_new


if __name__ == "__main__":
    df_done, done_set = load_done_set()
    print(f"Resuming: {len(done_set)} combinations already recorded.")

    records = []
    for data_set_index in DATASETS:
        for value_function_type in VALUE_FUNCTION_TYPES:
            for preferences_num in PREFERENCES_NUMS:
                for cognitive_biases in COGNITIVE_BIASES:
                    for random_index in range(N_INSTANCES):
                        key = (data_set_index, value_function_type, preferences_num,
                               cognitive_biases, random_index)
                        if key in done_set:
                            continue
                        
                        evalmatrix, preferences_observed, best_intervals_num = read_data_info(
                            data_set_index, value_function_type, preferences_num,
                            cognitive_biases, random_index,
                        )

                        # ---- SOR timing (hit-and-run sampling + inconsistency check) ----
                        t0 = time.time()
                        (_u_samples, min_incon_num, incon_check, incon_check_time
                         ) = u_sampler_SOR_for_time_test(
                            preferences_observed, evalmatrix, best_intervals_num, device,
                        )
                        sor_time = time.time() - t0

                        # ---- SGVI timing (Adam on ELBO) ----
                        t0 = time.time()
                        sgvi = SGVI(evalmatrix, best_intervals_num, preferences_observed,
                                    E, T_VI, W_G, device)
                        sgvi.sga()
                        sgvi_time = time.time() - t0

                        records.append({
                            "data_set_index": data_set_index,
                            "value_function_type": value_function_type,
                            "preferences_num": preferences_num,
                            "cognitive_biases": cognitive_biases,
                            "random_index": random_index,
                            "SGVI_time": sgvi_time,
                            "SOR_time": sor_time,
                            "min_incon_num": min_incon_num,
                            "incon_check": incon_check,
                            "incon_check_time": incon_check_time,
                        })
                        print(f"dataset{data_set_index}_{value_function_type}_"
                              f"{preferences_num}_{cognitive_biases}_random{random_index}: "
                              f"SGVI={sgvi_time:.3f}s, SOR={sor_time:.3f}s")

                        if len(records) >= 50:
                            df_done = flush(df_done, records)
                            records = []

    if records:
        flush(df_done, records)
