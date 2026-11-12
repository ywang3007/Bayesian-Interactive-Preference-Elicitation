"""
H_DVF heuristic runner with cognitive-bias noise injection.

Selects at each round the question with the largest min(p, 1-p), i.e. the pair
whose posterior-predictive preference is closest to 50/50. First round uses
the Dirichlet prior; later rounds use ``u_sampler`` on the biased reference.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
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

from SGVI import Disaggregation  # noqa: E402
from U_Sampler import u_sampler  # noqa: E402

from noise_utils import (  # noqa: E402
    apply_cognitive_bias,
    load_setting,
    sequence_output_path,
    timing_output_path,
)



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


def run_one_instance(args, best_intervals_num_set, device):
    setting_path = os.path.join(CODE_ROOT, "data", "simulated", f"simu_n={args.n}_m={args.m}_random{args.random_index}.json",
    )
    (
        evalmatrix,
        _preferences_all,
        preferences_non_domi,
        preferences_non_domi_index,
        value_true,
    ) = load_setting(setting_path, device)

    DM_answer = apply_cognitive_bias(preferences_non_domi, value_true, args.F3)
    all_action_set = get_all_action_set(evalmatrix, preferences_non_domi_index, device)
    action_num = all_action_set.shape[0]

    best_intervals_num = best_intervals_num_set[
        f"n{args.n}_m{args.m}_random{args.random_index}"
    ][0]
    intervals_num = best_intervals_num

    disaggregation = Disaggregation(evalmatrix, intervals_num, device)
    V = disaggregation.V
    samples_num = 10000

    candidate_choices = list(range(action_num))
    cumulative_choices = []
    round_times = []
    instance_tag = f"H_DVF F3={args.F3:.2f} rand={args.random_index}"
    for i in tqdm(range(args.T), desc=instance_tag, mininterval=1.0, ncols=80):
        t_round = time.time()
        minPWI_set = []
        if not cumulative_choices:
            phi = torch.ones(evalmatrix.shape[1] * intervals_num).to(device)
            u = torch.distributions.dirichlet.Dirichlet(phi ** 2).sample((samples_num,))
            value_u = u @ V
            for j in candidate_choices:
                v1 = value_u[:, all_action_set[j][0]]
                v2 = value_u[:, all_action_set[j][1]]
                p = float((v1 > v2).float().mean())
                minPWI_set.append(min(p, 1 - p))
        else:
            reference_true = DM_answer[cumulative_choices]
            u_samples = u_sampler(reference_true, evalmatrix, intervals_num, device)
            value_u = u_samples @ V
            for j in candidate_choices:
                v1 = value_u[:, all_action_set[j][0]]
                v2 = value_u[:, all_action_set[j][1]]
                p = float((v1 > v2).float().mean())
                minPWI_set.append(min(p, 1 - p))

        max_index = minPWI_set.index(max(minPWI_set))
        cumulative_choices.append(int(candidate_choices[max_index]))
        candidate_choices.pop(max_index)
        round_times.append(time.time() - t_round)

    return cumulative_choices, round_times


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--random_index", type=int, required=True)
    p.add_argument("--F3", type=float, required=True)
    p.add_argument("--gpu", type=int, default=-1)
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
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_timing", action="store_true",
                   help="Also save per-round wall-clock time")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    out_path = sequence_output_path(
        args.results_root, "H_DVF", args.n, args.m, args.T, args.random_index, args.F3
    )
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} already exists")
        return

    with open(args.best_intervals_json, "r") as fp:
        best_intervals_num_set = json.load(fp)

    t0 = time.time()
    cumulative, round_times = run_one_instance(args, best_intervals_num_set, device)
    elapsed = time.time() - t0

    torch.save(cumulative, out_path)
    if args.save_timing:
        timing_path = timing_output_path(
            args.results_root, "H_DVF", args.n, args.m, args.T,
            args.random_index, args.F3,
        )
        torch.save(round_times, timing_path)
    print(
        f"[done] method=H_DVF n={args.n} m={args.m} T={args.T} "
        f"random_index={args.random_index} F3={args.F3:.2f} "
        f"elapsed={elapsed:.1f}s "
        f"per_round={[round(t, 3) for t in round_times]} -> {out_path}"
    )


if __name__ == "__main__":
    main()
