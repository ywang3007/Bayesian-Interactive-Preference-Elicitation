"""
Evaluate question sequences produced under cognitive-bias noise.

For each (method, n, m, T, random_index, F3) combination:
  1. Load the setting and apply the same cognitive bias used during the run.
  2. Load the stored question sequence (flat list of action indices).
  3. Feed ``DM_answer[sequence[:T]]`` to SGVI to obtain the posterior over u.
  4. Compute var / fPWI / fRAI / entropy and append a row to the CSV.

By default only F3 > 0 is evaluated; the F3 = 0 baseline is available in
``strategies_202509/results.csv`` (not ``results_ASP.csv``) and is merged
downstream. Pass ``--F3s 0.0 ...`` to also evaluate F3=0 here if needed.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


sys.path.append(os.path.join(CODE_ROOT, "core"))

from SGVI import SGVI  # noqa: E402
from Measures import (  # noqa: E402
    fPWI_calculate,
    fRAI_calculate,
    var_calculate,
    entropy_calculate,
)

from noise_utils import (  # noqa: E402
    apply_cognitive_bias,
    load_setting,
    sequence_output_path,
)


METHODS = [
    "MCTS_Var", "MCTS_Ent",
    "H_PWI_d1", "H_PWI_d2", "H_RAI_d1", "H_RAI_d2",
    "H_DVF", "H_RAND",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_all_action_set(evalmatrix, preferences_non_domi_index, device):
    n = evalmatrix.shape[0]
    pair_num = int(n * (n - 1) / 2)
    all_pairs = torch.zeros((pair_num, 2), dtype=torch.int64).to(device)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs[k][0] = int(i)
            all_pairs[k][1] = int(j)
            k += 1
    return all_pairs[preferences_non_domi_index]


def sequence_file_for_eval(results_root, method, n, m, T, random_index, F3):
    """Return the sequence file to read. MCTS files are per-T; heuristic
    files are always stored at T=T_MAX (default 8) and sliced during eval."""
    if method.startswith("MCTS_"):
        T_file = T  # one file per T
    else:
        T_file = 8  # heuristics saved at the maximum T
    return sequence_output_path(results_root, method, n, m, T_file, random_index, F3)


def result_generate(DM_answer, node_sequence, all_action_set, evalmatrix,
                    intervals_num, e, T_sgvi, W_g, device, conv_num, samples_num):
    """Compute (var, fPWI, fRAI, entropy) given a flat sequence of indices."""
    idx = torch.tensor(list(node_sequence), dtype=torch.long, device=device)
    reference = DM_answer[idx]
    sgvi = SGVI(evalmatrix, intervals_num, reference, e, T_sgvi, W_g, device)
    V = sgvi.V
    phi_set, _, _ = sgvi.sga()

    if torch.isnan(phi_set).any().item():
        mask = torch.isnan(phi_set).any(dim=1)
        phi_set_process = phi_set[~mask]
        phi_set_conv = phi_set_process[-conv_num:]
        phi_conv_mean = phi_set_process[-1]
        print("[warn] NaN encountered in phi_set")
    else:
        phi_set_conv = phi_set[-conv_num:]
        phi_conv_mean = torch.mean(phi_set_conv, dim=0)

    u_samples = torch.distributions.dirichlet.Dirichlet(phi_conv_mean ** 2).sample((samples_num,))
    var = var_calculate(phi_set_conv)
    fPWI = fPWI_calculate(u_samples, V, all_action_set)
    fRAI = fRAI_calculate(u_samples, V, all_action_set)
    entropy = entropy_calculate(u_samples, V, all_action_set)
    return var.item(), fPWI.item(), fRAI.item(), entropy.item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--Ts", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--F3s", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    p.add_argument("--random_indices", type=int, nargs="+",
                   default=list(range(20)))
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--results_root",
        type=str,
        default=os.path.join(CODE_ROOT, "results", "section5", "noise"),
    )
    p.add_argument(
        "--best_intervals_json",
        type=str,
        default=os.path.join(CODE_ROOT, "data", "best_intervals_num.json"),
    )
    p.add_argument("--csv_out", type=str, default=None,
                   help="Default: <results_root>/eval_results.csv")
    p.add_argument("--flush_every", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csv_out = args.csv_out or os.path.join(args.results_root, "eval_results.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    columns = ["n", "m", "method", "T", "random_index", "F3",
               "var", "fPWI", "fRAI", "entropy"]
    if os.path.exists(csv_out) and os.path.getsize(csv_out) > 0:
        df_done = pd.read_csv(csv_out)
    else:
        df_done = pd.DataFrame(columns=columns)
    if len(df_done) > 0:
        done_keys = set(zip(df_done["n"], df_done["m"], df_done["method"],
                            df_done["T"], df_done["random_index"],
                            df_done["F3"].astype(float).round(2)))
    else:
        done_keys = set()

    with open(args.best_intervals_json, "r") as fp:
        best_intervals_num_set = json.load(fp)

    # SGVI hyper-parameters (match Section 4.1 in the paper)
    e = 0.00001        # convergence tolerance delta
    T_sgvi = 100       # max Adam iterations
    W_g = 10000        # MC samples for the stochastic gradient
    conv_num = 50      # window used to average phi
    samples_num = 10000  # posterior sample size (= W)

    set_seed(0)
    records = []
    total_t0 = time.time()

    outer = tqdm(
        total=len(args.F3s) * len(args.random_indices),
        desc="eval instances",
        mininterval=1.0,
        ncols=80,
    )
    for F3 in args.F3s:
        for random_index in args.random_indices:
            outer.update(1)
            outer.set_postfix_str(f"F3={F3:.2f} rand={random_index}")
            setting_path = os.path.join(CODE_ROOT, "data", "simulated", f"simu_n={args.n}_m={args.m}_random{random_index}.json",
            )
            (
                evalmatrix,
                _preferences_all,
                preferences_non_domi,
                preferences_non_domi_index,
                value_true,
            ) = load_setting(setting_path, device)
            DM_answer = apply_cognitive_bias(preferences_non_domi, value_true, F3)
            all_action_set = get_all_action_set(evalmatrix, preferences_non_domi_index, device)

            best_intervals_num = best_intervals_num_set[
                f"n{args.n}_m{args.m}_random{random_index}"
            ][0]
            intervals_num = best_intervals_num

            for method in args.methods:
                for T in args.Ts:
                    key = (args.n, args.m, method, T, random_index, round(F3, 2))
                    if key in done_keys:
                        continue
                    seq_path = sequence_file_for_eval(
                        args.results_root, method, args.n, args.m, T, random_index, F3
                    )
                    if not os.path.exists(seq_path):
                        print(f"[miss] {seq_path} not found, skip")
                        continue
                    full_seq = torch.load(seq_path)
                    # In case the file is a list-of-lists (legacy MCTS format),
                    # unwrap. Otherwise assume flat list.
                    if isinstance(full_seq, list) and full_seq and isinstance(full_seq[0], list):
                        full_seq = full_seq[-1]
                    sequence = list(full_seq)[:T]

                    try:
                        var, fPWI, fRAI, entropy = result_generate(
                            DM_answer, sequence, all_action_set, evalmatrix,
                            intervals_num, e, T_sgvi, W_g, device, conv_num, samples_num
                        )
                    except Exception as ex:
                        print(f"[err] {method} T={T} F3={F3} rand={random_index}: {ex}")
                        continue

                    records.append({
                        "n": args.n,
                        "m": args.m,
                        "method": method,
                        "T": T,
                        "random_index": random_index,
                        "F3": round(F3, 2),
                        "var": var,
                        "fPWI": fPWI,
                        "fRAI": fRAI,
                        "entropy": entropy,
                    })
                    print(
                        f"[eval] method={method} T={T} F3={F3:.2f} "
                        f"rand={random_index} -> var={var:.4e} fPWI={fPWI:.4e} "
                        f"fRAI={fRAI:.4e} entropy={entropy:.4e}"
                    )

                    if len(records) >= args.flush_every:
                        df_done = pd.concat([df_done, pd.DataFrame(records)], ignore_index=True)
                        df_done.to_csv(csv_out, index=False)
                        records = []

    outer.close()
    if records:
        df_done = pd.concat([df_done, pd.DataFrame(records)], ignore_index=True)
        df_done.to_csv(csv_out, index=False)

    print(f"[all done] total elapsed {time.time() - total_t0:.1f}s; csv={csv_out}")


if __name__ == "__main__":
    main()
