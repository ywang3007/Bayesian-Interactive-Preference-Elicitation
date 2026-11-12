"""Evaluate Section 5 question sequences (main + C_p sensitivity).

Reads the question sequences produced by the MCTS and heuristic runners,
re-fits an SGVI posterior conditioned on the DM's true preferences along
that sequence, and computes 4 uncertainty metrics per instance:
``var``, ``fPWI``, ``fRAI``, ``entropy``.

Covers 14 method tags:
  - MCTS_Var / MCTS_Entropy_new (legacy Cp=1/sqrt(2) naming, kept for
    compatibility with the main-paper plots)
  - MCTS_Var_Cp_{0.2, 0.5, 0.7071067811865475, 1.0}         (Appendix I)
  - MCTS_Entropy_Cp_{0.2, 0.5, 0.7071067811865475, 1.0}     (Appendix I)
  - E_PWI_d1, E_PWI_d2, E_RAI_d1, E_RAI_d2 (heuristic depth-1/2)
  - DVF, RAND baselines

Output: ``results/section5/results.csv`` (resumable; skips already-recorded rows).
"""

import sys
import os
import json
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
from Measures import (
    var_calculate, fPWI_calculate, fRAI_calculate, entropy_calculate,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------- experimental grid ----------
PROBLEM_SIZES = [(6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
                 (8, 2), (8, 4), (8, 5)]
# Cp values used for the Appendix I sensitivity sweep. "0.7071..." is the
# baseline value 1/sqrt(2), kept with its full literal so the saved folder
# tag ``Cp0.7071067811865475`` round-trips with what MCTS.py writes.
CP_SWEEP_VALUES = ["0.2", "0.5", "0.7071067811865475", "1.0"]

METHODS = [
    'MCTS_Var',
    'MCTS_Entropy_new',
    'E_RAI_d1', 'E_RAI_d2', 'E_PWI_d1', 'E_PWI_d2',
    'DVF', 'RAND',
] + [f'MCTS_Var_Cp_{cp}' for cp in CP_SWEEP_VALUES] \
  + [f'MCTS_Entropy_Cp_{cp}' for cp in CP_SWEEP_VALUES]
ROUNDS_LIST = [2, 4, 6, 8]
N_INSTANCES = 20
COMPUTATION_BUDGET = 300

# ---------- SGVI parameters (match Section 4.1 in the paper) ----------
E = 1e-5            # convergence tolerance delta
T_VI = 100          # max Adam iterations
W_G = 10000         # MC samples for the stochastic gradient
CONV_NUM = 50       # window (last iterations) used to average phi
SAMPLES_NUM = 10000 # posterior sample size for the metric estimators (= W)

# ---------- I/O paths ----------
SETTINGS_DIR = f"{CODE_ROOT}/data/simulated"
BEST_GAMMA_JSON = f"{CODE_ROOT}/data/best_intervals_num.json"
CSV_PATH = f"{CODE_ROOT}/results/section5/results.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_all_action_set(evalmatrix, preferences_non_domi_index):
    n = evalmatrix.shape[0]
    pair_num = int(n * (n - 1) / 2)
    pairs = torch.zeros((pair_num, 2), dtype=torch.int64).to(device)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs[k][0], pairs[k][1] = int(i), int(j)
            k += 1
    return pairs[preferences_non_domi_index]


def read_data_info(n, m, random_index):
    with open(f"{SETTINGS_DIR}/simu_n={n}_m={m}_random{random_index}.json") as fp:
        setting = json.load(fp)
    evalmatrix = torch.tensor(setting['evalmatrix']).to(device)
    preferences_non_domi = torch.tensor(setting['preferences_non_domi']).to(device)
    preferences_non_domi_index = torch.tensor(setting['preferences_non_domi_index']).to(device)
    all_action_set = get_all_action_set(evalmatrix, preferences_non_domi_index)

    with open(BEST_GAMMA_JSON) as fp:
        best_gamma = json.load(fp)
    intervals_num = best_gamma[f"n{n}_m{m}_random{random_index}"][0]

    return preferences_non_domi, all_action_set, evalmatrix, intervals_num


MCTS_CP_PREFIXES = {
    "MCTS_Var_Cp_":     "",         # subdir "MCTS_result_simudata_Cp{cp}"
    "MCTS_Entropy_Cp_": "Entropy_",  # subdir "MCTS_result_simudata_Entropy_Cp{cp}"
}


def get_node_sequence(method_name, n, m, random_index, max_round_number, cb):
    """Load the question-sequence .pt file produced by the corresponding runner."""
    seq_root = f"{CODE_ROOT}/results/section5_sequences"
    heur_root = f"{CODE_ROOT}/results/section5_heuristic_sequences"

    if method_name == 'MCTS_Var':
        p = (f"{seq_root}/MCTS_result_simudata_Cp0.7071067811865475"
             f"/new_node_sequence_n{n}_m{m}_random{random_index}"
             f"_MAX{max_round_number}_BUDGET{cb}.pt")
        return torch.load(p)[-1]
    if method_name == 'MCTS_Entropy_new':
        p = (f"{seq_root}/MCTS_result_simudata_Entropy_Cp0.7071067811865475"
             f"/new_node_sequence_n{n}_m{m}_random{random_index}"
             f"_MAX{max_round_number}_BUDGET{cb}.pt")
        return torch.load(p)[-1]
    for prefix, subdir_tag in MCTS_CP_PREFIXES.items():
        if method_name.startswith(prefix):
            cp = method_name[len(prefix):]
            p = (f"{seq_root}/MCTS_result_simudata_{subdir_tag}Cp{cp}"
                 f"/new_node_sequence_n{n}_m{m}_random{random_index}"
                 f"_MAX{max_round_number}_BUDGET{cb}.pt")
            seq = torch.load(p)
            # Legacy baseline (Cp=0.7071) was saved as list-of-lists; sweep
            # runs are plain tensors / lists.
            if isinstance(seq, list) and seq and isinstance(seq[0], list):
                seq = seq[-1]
            return seq.tolist() if hasattr(seq, "tolist") else list(seq)
    if method_name in {'E_RAI_d1', 'E_RAI_d2', 'E_PWI_d1', 'E_PWI_d2'}:
        p = (f"{heur_root}/H_O_X_d/{method_name}_node_sequence"
             f"_n{n}_m{m}_MAX10_random{random_index}.pt")
        return torch.load(p)[:max_round_number]
    if method_name == 'DVF':
        p = f"{heur_root}/H_DVF/DVF_node_sequence_n{n}_m{m}_random{random_index}.pt"
        return torch.load(p)[:max_round_number]
    if method_name == 'RAND':
        p = f"{heur_root}/H_RAND/RAND_node_sequence_n{n}_m{m}_random{random_index}.pt"
        return torch.load(p)[:max_round_number]
    raise ValueError(f"unknown method: {method_name}")


def evaluate_one(DM_answer, node_sequence, all_action_set, evalmatrix, intervals_num):
    reference = DM_answer[node_sequence]
    sgvi = SGVI(evalmatrix, intervals_num, reference, E, T_VI, W_G, device)
    V = sgvi.V
    phi_set, _, _ = sgvi.sga()

    if torch.isnan(phi_set).any().item():
        mask = torch.isnan(phi_set).any(dim=1)
        phi_set_process = phi_set[~mask]
        phi_set_conv = phi_set_process[-CONV_NUM:]
        phi_conv_mean = phi_set_process[-1]
    else:
        phi_set_conv = phi_set[-CONV_NUM:]
        phi_conv_mean = torch.mean(phi_set_conv, dim=0)

    u_samples = torch.distributions.dirichlet.Dirichlet(phi_conv_mean ** 2).sample((SAMPLES_NUM,))
    var = var_calculate(phi_set_conv).item()
    fPWI = fPWI_calculate(u_samples, V, all_action_set).item()
    fRAI = fRAI_calculate(u_samples, V, all_action_set).item()
    entropy = entropy_calculate(u_samples, V, all_action_set).item()
    return var, fPWI, fRAI, entropy


def load_done_set():
    columns = ["n", "m", "method", "rounds", "random_index",
               "var", "fPWI", "fRAI", "entropy"]
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=columns)
    done = set(zip(df["n"], df["m"], df["method"], df["rounds"], df["random_index"]))
    return df, done


def flush(df_old, records):
    df_new = pd.concat([df_old, pd.DataFrame(records)], ignore_index=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_new.to_csv(CSV_PATH, index=False)
    return df_new


if __name__ == "__main__":
    set_seed(0)
    df_done, done_set = load_done_set()
    print(f"Resuming: {len(done_set)} rows already recorded.")

    records = []
    for n, m in PROBLEM_SIZES:
        for method in METHODS:
            for max_round in ROUNDS_LIST:
                for random_index in range(N_INSTANCES):
                    key = (n, m, method, max_round, random_index)
                    if key in done_set:
                        continue

                    DM_answer, all_action_set, evalmatrix, intervals_num = read_data_info(
                        n, m, random_index,
                    )
                    node_sequence = get_node_sequence(
                        method, n, m, random_index, max_round, COMPUTATION_BUDGET,
                    )
                    var, fPWI, fRAI, entropy = evaluate_one(
                        DM_answer, node_sequence, all_action_set, evalmatrix, intervals_num,
                    )
                    print(f"n={n} m={m} method={method} rounds={max_round} "
                          f"random={random_index} -> var={var:.4f} fPWI={fPWI:.4f} "
                          f"fRAI={fRAI:.4f} entropy={entropy:.4f}")

                    records.append({
                        "n": n, "m": m, "method": method,
                        "rounds": max_round, "random_index": random_index,
                        "var": var, "fPWI": fPWI, "fRAI": fRAI, "entropy": entropy,
                    })
                    if len(records) >= 50:
                        df_done = flush(df_done, records)
                        records = []

    if records:
        flush(df_done, records)
