"""
H_RAND runner with cognitive-bias "awareness".

The random policy does not consult the DM when selecting questions, so the
question sequence is identical across F3 levels. The script therefore
produces the SAME question sequence for F3 = 0 / 0.1 / 0.2 / 0.3 at a given
(random_index, T). We still write one file per F3 to keep the output layout
uniform; the differences between F3 levels show up during evaluation only.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


sys.path.append(os.path.join(CODE_ROOT, "core"))
from noise_utils import (  # noqa: E402
    load_setting,
    sequence_output_path,
    timing_output_path,
)



def get_action_num(evalmatrix, preferences_non_domi_index):
    n = evalmatrix.shape[0]
    pair_num = int(n * (n - 1) / 2)
    kept = preferences_non_domi_index
    if torch.is_tensor(kept):
        return int(kept.shape[0])
    return pair_num  # fallback (should not happen in our settings)


def run_one_instance(args, device):
    setting_path = os.path.join(CODE_ROOT, "data", "simulated", f"simu_n={args.n}_m={args.m}_random{args.random_index}.json",
    )
    (
        evalmatrix,
        _preferences_all,
        _preferences_non_domi,
        preferences_non_domi_index,
        _value_true,
    ) = load_setting(setting_path, device)

    action_num = get_action_num(evalmatrix, preferences_non_domi_index)
    candidate_choices = list(range(action_num))
    round_times = []
    cumulative_choices = []
    for _ in range(args.T):
        t0 = time.time()
        pick = random.choice(candidate_choices)
        candidate_choices.remove(pick)
        cumulative_choices.append(int(pick))
        round_times.append(time.time() - t0)
    return cumulative_choices, round_times


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--random_index", type=int, required=True)
    p.add_argument("--F3", type=float, required=True)
    p.add_argument(
        "--results_root",
        type=str,
        default=os.path.join(CODE_ROOT, "results", "section5", "noise"),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_timing", action="store_true",
                   help="Also save per-round wall-clock time")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    out_path = sequence_output_path(
        args.results_root, "H_RAND", args.n, args.m, args.T, args.random_index, args.F3
    )
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} already exists")
        return

    t0 = time.time()
    cumulative, round_times = run_one_instance(args, device)
    elapsed = time.time() - t0

    torch.save(cumulative, out_path)
    if args.save_timing:
        timing_path = timing_output_path(
            args.results_root, "H_RAND", args.n, args.m, args.T,
            args.random_index, args.F3,
        )
        torch.save(round_times, timing_path)
    print(
        f"[done] method=H_RAND n={args.n} m={args.m} T={args.T} "
        f"random_index={args.random_index} F3={args.F3:.2f} "
        f"elapsed={elapsed:.1f}s -> {out_path}"
    )


if __name__ == "__main__":
    main()
