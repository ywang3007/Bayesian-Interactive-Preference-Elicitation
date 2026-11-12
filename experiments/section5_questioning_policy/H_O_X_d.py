"""
H_O_X_d heuristic runner with cognitive-bias noise injection.

Covers four methods via CLI:
    --measure PWI --depth 1   -> H_PWI_d1
    --measure PWI --depth 2   -> H_PWI_d2
    --measure RAI --depth 1   -> H_RAI_d1
    --measure RAI --depth 2   -> H_RAI_d2

Each run selects questions for a single instance (random_index) and saves the
full question sequence of length T (default 8) as a flat Python list of
action indices. The evaluation script slices this list for shorter T values.

The core recursion ``v(...)`` and operator ``O(...)`` are copied verbatim from
``H_O_X_d.py`` so the behavior is identical to the F3=0 baseline, except
that ``DM_answer`` is biased via :func:`apply_cognitive_bias`.
"""

import argparse
import json
import math
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


# ---------- path setup (env-var portable) ----------
sys.path.append(os.path.join(CODE_ROOT, "core"))

from SGVI import Disaggregation  # noqa: E402
from Measures import (  # noqa: E402
    fPWI_calculate,
    fQUE_calculate,
    fRAI_calculate,
    NEC_test,
)
from U_Sampler import u_sampler  # noqa: E402

from noise_utils import (  # noqa: E402
    apply_cognitive_bias,
    load_setting,
    sequence_output_path,
    timing_output_path,
)


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


def O(x, y, operator, PWI1, PWI2):
    if operator == "W":
        return min(x, y)
    if operator == "M":
        return (x + y) / 2
    if operator == "E":
        return (PWI1 * x + PWI2 * y) / (PWI1 + PWI2)
    raise ValueError(operator)


def f_tilde(measure, preferences_observed, evalmatrix, V, intervals_num, all_action_set, t, device):
    u_samples = u_sampler(preferences_observed, evalmatrix, intervals_num, device)
    if measure == "QUE":
        return fQUE_calculate(preferences_observed, V, all_action_set) + t
    if measure == "PWI":
        return fPWI_calculate(u_samples, V, all_action_set)
    if measure == "RAI":
        return fRAI_calculate(u_samples, V, all_action_set)
    raise ValueError(measure)


def Q_generate(all_action_set, S):
    delete_index = []
    for i in range(all_action_set.shape[0]):
        for j in range(S.shape[0]):
            if (all_action_set[i][0] == S[j][0] and all_action_set[i][1] == S[j][1]) or (
                all_action_set[i][0] == S[j][1] and all_action_set[i][1] == S[j][0]
            ):
                delete_index.append(i)
    kept = [x for x in range(all_action_set.shape[0]) if x not in delete_index]
    return all_action_set[kept]


def action_index(all_action_set, q):
    for i in range(all_action_set.shape[0]):
        a = all_action_set[i]
        if (a[0] == q[0] and a[1] == q[1]) or (a[0] == q[1] and a[1] == q[0]):
            return i
    return None


def v(S, operator, measure, t, evalmatrix, intervals_num, all_action_set, V, d, device):
    if t == 0:
        return -f_tilde(
            measure, S, evalmatrix, V, intervals_num, all_action_set, d - t, device
        )
    Q = Q_generate(all_action_set, S)
    u_samples = u_sampler(S, evalmatrix, intervals_num, device)
    value_u = u_samples @ V

    comparisons = []
    for q in Q:
        relation_test = q
        NEC1 = NEC_test(S, V, relation_test)
        NEC2 = NEC_test(S, V, relation_test[[1, 0]])

        i = action_index(all_action_set, q)
        value1 = value_u[:, all_action_set[i][0]]
        value2 = value_u[:, all_action_set[i][1]]
        p = float((value1 > value2).float().mean())
        PWI1, PWI2 = p, 1 - p
        S_new1 = torch.cat((S, q.unsqueeze(0)), dim=0)
        S_new2 = torch.cat((S, q[[1, 0]].unsqueeze(0)), dim=0)

        if operator == "E" and NEC1 is True:
            v1 = v(S_new1, operator, measure, t - 1, evalmatrix, intervals_num, all_action_set, V, d, device)
            v2 = 0
        elif operator == "E" and NEC2 is True:
            v1 = 0
            v2 = v(S_new2, operator, measure, t - 1, evalmatrix, intervals_num, all_action_set, V, d, device)
        else:
            v1 = v(S_new1, operator, measure, t - 1, evalmatrix, intervals_num, all_action_set, V, d, device)
            v2 = v(S_new2, operator, measure, t - 1, evalmatrix, intervals_num, all_action_set, V, d, device)
        comparisons.append(O(v1, v2, operator, PWI1, PWI2))
    return max(comparisons)


def run_one_instance(args, best_intervals_num_set, device):
    set_seed(args.random_index)
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
    if args.n == 8 and args.m == 2 and intervals_num == 1:
        # Original heuristics skip this corner case (volesti hit-and-run has
        # issues when the feasible polytope is a 1D interval).
        print(f"[skip] n=8, m=2 with intervals_num=1 is unsupported "
              f"(random_index={args.random_index}, F3={args.F3})")
        return None, None

    disaggregation = Disaggregation(evalmatrix, intervals_num, device)
    V = disaggregation.V

    operator = "E"
    measure = args.measure
    d = args.depth

    candidate_choices = list(range(action_num))
    cumulative_choices = []
    round_times = []
    instance_tag = (
        f"H_{args.measure}_d{args.depth} F3={args.F3:.2f} rand={args.random_index}"
    )
    round_iter = tqdm(
        range(args.T), desc=instance_tag, mininterval=1.0, ncols=80
    )
    for i in round_iter:
        t_round = time.time()
        comparisons_set = []
        inner = tqdm(
            candidate_choices,
            desc=f"  round {i + 1}/{args.T}",
            leave=False,
            mininterval=1.0,
            ncols=80,
        )
        for j in inner:
            select = cumulative_choices.copy()
            reference_true = DM_answer[select]
            u_samples = u_sampler(reference_true, evalmatrix, intervals_num, device)
            value_u = u_samples @ V

            relation_test = all_action_set[j]
            NEC1 = NEC_test(reference_true, V, relation_test)
            NEC2 = NEC_test(reference_true, V, relation_test[[1, 0]])

            value1 = value_u[:, all_action_set[j][0]]
            value2 = value_u[:, all_action_set[j][1]]
            p = float((value1 > value2).float().mean())
            PWI1, PWI2 = p, 1 - p

            if not cumulative_choices:
                S1 = all_action_set[j].unsqueeze(0)
                S2 = all_action_set[j][[1, 0]].unsqueeze(0)
            else:
                S1 = torch.cat((reference_true, all_action_set[j].unsqueeze(0)), dim=0)
                S2 = torch.cat((reference_true, all_action_set[j][[1, 0]].unsqueeze(0)), dim=0)

            if operator == "E" and NEC1 is True:
                v1 = v(S1, operator, measure, d - 1, evalmatrix, intervals_num,
                       all_action_set, V, d, device)
                v2 = 0
            elif operator == "E" and NEC2 is True:
                v1 = 0
                v2 = v(S2, operator, measure, d - 1, evalmatrix, intervals_num,
                       all_action_set, V, d, device)
            else:
                v1 = v(S1, operator, measure, d - 1, evalmatrix, intervals_num,
                       all_action_set, V, d, device)
                v2 = v(S2, operator, measure, d - 1, evalmatrix, intervals_num,
                       all_action_set, V, d, device)
            comparisons_set.append(O(v1, v2, operator, PWI1, PWI2))

        inner.close()
        max_index = comparisons_set.index(max(comparisons_set))
        cumulative_choices.append(int(candidate_choices[max_index]))
        candidate_choices.pop(max_index)
        round_times.append(time.time() - t_round)
    round_iter.close()

    return cumulative_choices, round_times


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--measure", choices=["PWI", "RAI"], required=True)
    p.add_argument("--depth", type=int, choices=[1, 2], required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--T", type=int, default=8,
                   help="Number of rounds to run; heuristics only need the max T")
    p.add_argument("--random_index", type=int, required=True)
    p.add_argument("--F3", type=float, required=True)
    p.add_argument("--gpu", type=int, default=-1,
                   help="Logical GPU index; -1 uses CPU (default, matches original)")
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

    method_tag = f"H_{args.measure}_d{args.depth}"
    out_path = sequence_output_path(
        args.results_root, method_tag, args.n, args.m, args.T, args.random_index, args.F3
    )
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} already exists")
        return

    with open(args.best_intervals_json, "r") as fp:
        best_intervals_num_set = json.load(fp)

    t0 = time.time()
    cumulative, round_times = run_one_instance(args, best_intervals_num_set, device)
    elapsed = time.time() - t0

    if cumulative is None:
        # Instance was intentionally skipped; do not write any output so a
        # future retry can pick it up cleanly.
        return

    torch.save(cumulative, out_path)
    if args.save_timing:
        timing_path = timing_output_path(
            args.results_root, method_tag, args.n, args.m, args.T,
            args.random_index, args.F3,
        )
        torch.save(round_times, timing_path)
    print(
        f"[done] method={method_tag} n={args.n} m={args.m} T={args.T} "
        f"random_index={args.random_index} F3={args.F3:.2f} "
        f"elapsed={elapsed:.1f}s "
        f"per_round={[round(t, 2) for t in round_times]} -> {out_path}"
    )


if __name__ == "__main__":
    main()
