"""Compute per-instance ASP for the proposed SGVI method and SOR.

Iterates over every (dataset, value-function type, preferences_num, bias,
random_index) combination; for each one loads the SGVI inference output and
the SOR hit-and-run samples, evaluates ASP against the ground-truth
preference ordering, and writes one row per instance to:

    results/section4/ASP_results.csv

Prerequisite inputs (all under ``results/section4_*/``):
    results/section4_sgvi_outputs/SGVIresults_dataset*.pt   (from run_inference.py)
    results/section4_sor_samples/samples_dataset*.pt        (from SOR sampler)

Downstream consumer:
    summarize_ASP.py   (produces ASP_table.csv + Wilcoxon stats from this CSV)
"""

import sys
import os
import json

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
from Measures import ASP_calculate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------- experimental grid (matches the shipped CSV) ----------
DATASETS = range(8)
VALUE_FUNCTION_TYPES = ['linear', 'concave', 'convex', 'concave&convex']
PREFERENCES_NUMS = [5, 10, 15, 20, 30, 40, 60, 80]
COGNITIVE_BIASES = [0, 0.1, 0.2, 0.3]
N_INSTANCES = 20

# ---------- ASP evaluation parameters ----------
CONV_LAST_N = 50           # last-N phi iterates averaged for the posterior mean
U_SAMPLES = 10000          # draws from the Dirichlet posterior
SOR_SAMPLE_CAP = 50000     # concatenate SOR sample files up to this total

# ---------- I/O paths ----------
SETTINGS_DIR    = f"{CODE_ROOT}/data/section4_settings"
SGVI_OUT_DIR    = f"{CODE_ROOT}/results/section4_sgvi_outputs"
SOR_SAMPLES_DIR = f"{CODE_ROOT}/results/section4_sor_samples"
ASP_CSV_PATH    = f"{CODE_ROOT}/results/section4/ASP_results.csv"


def _posterior_mean_phi(phi_set):
    """Average the last ``CONV_LAST_N`` phi iterates; if NaN appeared mid-run,
    fall back to the last pre-NaN iterate."""
    if torch.isnan(phi_set).any():
        nan_row = torch.nonzero(torch.isnan(phi_set).sum(dim=1))[0].item()
        return phi_set[nan_row - 1].to(device)
    return torch.mean(phi_set[-CONV_LAST_N:], dim=0).to(device)


def _load_sor_samples(data_set_index, vft, pref_num, bias, random_index, best_intervals_num):
    """Concatenate SOR hit-and-run sample files, capped at ``SOR_SAMPLE_CAP``."""
    samples = None
    i = 0
    while True:
        p = (f"{SOR_SAMPLES_DIR}/samples_dataset{data_set_index}_{vft}_{pref_num}"
             f"_{bias:f}_random{random_index}_intervalsnum{best_intervals_num}_index{i}.pt")
        if not os.path.exists(p):
            break
        chunk = torch.load(p).to(device)
        samples = chunk if samples is None else torch.cat((samples, chunk), dim=0)
        i += 1
        if samples.shape[0] >= SOR_SAMPLE_CAP:
            break
    return samples


def _compute_asp_for_instance(data_set_index, vft, pref_num, bias, random_index):
    """Return (ASP_SGVI, ASP_SOR) for one instance."""
    settings_path = (f"{SETTINGS_DIR}/settings_dataset{data_set_index}_{vft}_{pref_num}"
                     f"_{bias:f}_random{random_index}.json")
    with open(settings_path, 'r') as fp:
        setting = json.load(fp)
    preferences_all = torch.tensor(setting['preferences_all']).to(device)

    # SGVI side
    sgvi_path = (f"{SGVI_OUT_DIR}/SGVIresults_dataset{data_set_index}_{vft}_{pref_num}"
                 f"_{bias:f}_random{random_index}.pt")
    sgvi_result = torch.load(sgvi_path)
    best_intervals_num = sgvi_result['best_intervals_num']
    V = sgvi_result['V_all'][best_intervals_num - 1].to(device)
    phi_set = sgvi_result['phi_set_all'][best_intervals_num - 1]
    phi = _posterior_mean_phi(phi_set)
    u_sgvi = torch.distributions.dirichlet.Dirichlet(phi ** 2).sample((U_SAMPLES,))
    asp_sgvi = ASP_calculate(u_sgvi, V, preferences_all).item()

    # SOR side
    u_sor = _load_sor_samples(data_set_index, vft, pref_num, bias, random_index, best_intervals_num)
    asp_sor = ASP_calculate(u_sor, V, preferences_all).item()

    return asp_sgvi, asp_sor


if __name__ == "__main__":
    rows = []
    for data_set_index in DATASETS:
        for vft in VALUE_FUNCTION_TYPES:
            for pref_num in PREFERENCES_NUMS:
                for bias in COGNITIVE_BIASES:
                    for random_index in range(N_INSTANCES):
                        asp_sgvi, asp_sor = _compute_asp_for_instance(
                            data_set_index, vft, pref_num, bias, random_index,
                        )
                        rows.append({
                            "data_set_index": data_set_index,
                            "value_function_type": vft,
                            "preferences_num": pref_num,
                            "cognitive_biases": bias,
                            "random_index": random_index,
                            "ASP_SGVI_repara": asp_sgvi,
                            "ASP_SOR": asp_sor,
                        })
                    print(f"dataset{data_set_index}_{vft}_{pref_num}_{bias}: done")

    os.makedirs(os.path.dirname(ASP_CSV_PATH), exist_ok=True)
    pd.DataFrame(rows).to_csv(ASP_CSV_PATH, index=False)
    print(f"[saved] {ASP_CSV_PATH}  ({len(rows)} rows)")
