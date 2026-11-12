import sys

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----

sys.path.append(f'{CODE_ROOT}/core')
sys.path.append(f'{CODE_ROOT}')
sys.path.append(f'{CODE_ROOT}/core')
sys.path.append(f'{CODE_ROOT}/experiments')
sys.path.append(f'{CODE_ROOT}/core')
import torch
import os
import json
from NpEncoder import NpEncoder
import time
import random
import numpy as np
import pandas as pd
from SGVI import SGVI
from Measures import fRAI_calculate
from Measures import fPWI_calculate
from Measures import var_calculate
from Measures import entropy_calculate
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=10)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_data_info(n, m, random_index):
    setting_path = f'{CODE_ROOT}/data/simulated/simu_n={n}_m={m}_random{random_index}.json'
    with open(setting_path, 'r') as fp:
        setting = json.load(fp)
    evalmatrix = torch.tensor(setting['evalmatrix']).to(device)
    preferences_non_domi = torch.tensor(setting['preferences_non_domi']).to(device)
    preferences_non_domi_index = torch.tensor(setting['preferences_non_domi_index']).to(device)
    DM_answer = preferences_non_domi
    all_action_set = get_all_action_set(evalmatrix, preferences_non_domi_index)

    with open(f'{CODE_ROOT}/data/best_intervals_num.json', 'r') as fp:
        best_intervals_num_set = json.load(fp)
    best_intervals_num = best_intervals_num_set[f'n{n}_m{m}_random{random_index}'][0]
    intervals_num = best_intervals_num
    return DM_answer, all_action_set, evalmatrix, intervals_num

def get_node_sequence(method_name, n, m, random_index, max_round_number, computation_budget):
    if method_name == 'MCTS_Var':
        file_path = f'{CODE_ROOT}/results/section5_sequences/MCTS_result_simudata_Cp0.7071067811865475/new_node_sequence_n{n}_m{m}_random{random_index}_MAX{max_round_number}_BUDGET{computation_budget}.pt'
        node_sequence = torch.load(file_path)[-1]
    elif method_name == 'MCTS_Entropy':
        file_path = f'{CODE_ROOT}/results/section5_sequences/MCTS_result_simudata_Entropy/new_node_sequence_n{n}_m{m}_random{random_index}_MAX{max_round_number}_BUDGET{computation_budget}.pt'
        node_sequence = torch.load(file_path)[-1]
    elif method_name == 'E_RAI_d1':
        file_path = f'experiments/Heuristics/H_result/H_O_X_d/E_RAI_d1_node_sequence_n{n}_m{m}_MAX10_random{random_index}.pt'
        node_sequence = torch.load(file_path)[:MAX_ROUND_NUMBER]
    elif method_name == 'E_RAI_d2':
        file_path = f'experiments/Heuristics/H_result/H_O_X_d/E_RAI_d2_node_sequence_n{n}_m{m}_MAX10_random{random_index}.pt'
        node_sequence = torch.load(file_path)[:MAX_ROUND_NUMBER]
    elif method_name == 'E_PWI_d1':
        file_path = f'experiments/Heuristics/H_result/H_O_X_d/E_PWI_d1_node_sequence_n{n}_m{m}_MAX10_random{random_index}.pt'
        node_sequence = torch.load(file_path)[:MAX_ROUND_NUMBER]
    elif method_name == 'E_PWI_d2':
        file_path = f'experiments/Heuristics/H_result/H_O_X_d/E_PWI_d2_node_sequence_n{n}_m{m}_MAX10_random{random_index}.pt'
        node_sequence = torch.load(file_path)[:MAX_ROUND_NUMBER]
    elif method_name == 'DVF':
        file_path = f'experiments/Heuristics/H_result/H_DVF/DVF_node_sequence_n{n}_m{m}_random{random_index}.pt'
        node_sequence = torch.load(file_path)[:MAX_ROUND_NUMBER]
    elif method_name == 'RAND':
        file_path = f'experiments/Heuristics/H_result/H_RAND/RAND_node_sequence_n{n}_m{m}_random{random_index}.pt'
        node_sequence = torch.load(file_path)[:MAX_ROUND_NUMBER]
    
    return node_sequence

def get_all_action_set(evalmatrix, preferences_non_domi_index):
    # Build the full set of pairwise questions (n choose 2) and keep only
    # the non-dominated pairs (a vs. b).
    n = evalmatrix.shape[0]
    action_num = int(n * (n-1) / 2)
    all_action_set = torch.zeros((action_num, 2), dtype=torch.int64).to(device)
    k = 0
    for i in range(n):
        for j in range(i +1, n):
            all_action_set[k][0] = int(i)
            all_action_set[k][1] = int(j)
            k += 1
    all_action_set = all_action_set[preferences_non_domi_index]
    return all_action_set

def result_generate(DM_answer, node_sequence, all_action_set, evalmatrix, intervals_num, e, T, W_g, device, conv_num, samples_num):
    reference = DM_answer[node_sequence]
    sgvi = SGVI(evalmatrix, intervals_num, reference, e, T, W_g, device)
    V = sgvi.V
    phi_set, _, _ = sgvi.sga()
    
    if torch.isnan(phi_set).any().item():
        # Drop SGVI iterations whose phi contains NaNs, then re-take the
        # last conv_num rows as the converged window.
        mask = torch.isnan(phi_set).any(dim=1)
        phi_set_process = phi_set[~mask]
        phi_set_conv = phi_set_process[-conv_num:]
        phi_conv_mean = phi_set_process[-1]
        print("NaN encountered in phi_set.")
    else:
        phi_set_conv = phi_set[-conv_num:]
        phi_conv_mean = torch.mean(phi_set_conv, dim=0)
    u_samples = torch.distributions.dirichlet.Dirichlet(phi_conv_mean ** 2).sample((samples_num,))
    var = var_calculate(phi_set_conv) 
    fPWI = fPWI_calculate(u_samples, V, all_action_set)
    fRAI = fRAI_calculate(u_samples, V, all_action_set)
    entropy = entropy_calculate(u_samples, V, all_action_set)
    return var.item(), fPWI.item(), fRAI.item(), entropy.item()


set_seed(0)
N_INSTANCES = 5  # 5 random instances per (method, T, CB) cell
for method in ['MCTS_Entropy', 'MCTS_Var']:
    records = []  # per-method: each method gets its own CSV
    for MAX_ROUND_NUMBER in [2, 4, 6, 8]:
        for n, m in [(8,3)]:
            for random_index in range(N_INSTANCES):
                for COMPUTATION_BUDGET in [100, 200, 300, 400, 500]:
                    # ---- load instance ----
                    DM_answer, all_action_set, evalmatrix, intervals_num = read_data_info(n, m, random_index)

                    # SGVI hyper-parameters (match Section 4.1 in the paper)
                    e = 0.00001        # convergence tolerance delta
                    T = 100            # max Adam iterations
                    W_g = 10000        # MC samples for the stochastic gradient
                    conv_num = 50      # window used to average phi
                    samples_num = 10000  # posterior sample size (= W)

                    # ---- run experiment ----
                    _seq_subdir = "Entropy" if method == "MCTS_Entropy" else "Var"
                    node_sequence_path = f'{CODE_ROOT}/results/section5_CB_sequences/{_seq_subdir}/node_sequence_n{n}_m{m}_random{random_index}_MAX{MAX_ROUND_NUMBER}_BUDGET{COMPUTATION_BUDGET}.pt'
                    node_sequence = torch.load(node_sequence_path)[-1]
                    var, fPWI, fRAI, entropy = result_generate(
                        DM_answer, node_sequence, all_action_set, evalmatrix,
                        intervals_num, e, T, W_g, device, conv_num, samples_num
                    )

                    print(f'n={n}, m={m}, method={method}, rounds={MAX_ROUND_NUMBER}, CB={COMPUTATION_BUDGET}, random_index={random_index} -> var: {var}, fPWI: {fPWI}, fRAI: {fRAI}, entropy: {entropy}')

                    # ---- append result ----
                    records.append({
                        "n": n,
                        "m": m,
                        "method": method,
                        "rounds": MAX_ROUND_NUMBER,
                        "CB": COMPUTATION_BUDGET,
                        "random_index": random_index,
                        "var": var.item() if torch.is_tensor(var) else var,
                        "fPWI": fPWI.item() if torch.is_tensor(fPWI) else fPWI,
                        "fRAI": fRAI.item() if torch.is_tensor(fRAI) else fRAI,
                        "entropy": entropy.item() if torch.is_tensor(entropy) else entropy
                    })

                    _csv_tag = "MCTS_Ent" if method == "MCTS_Entropy" else "MCTS_Var"
                    df = pd.DataFrame(records)
                    df.to_csv(f"{CODE_ROOT}/results/section5/results_CBtest_{_csv_tag}.csv", index=False)

