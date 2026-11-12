"""One-shot script to generate synthetic DM preference data for the
question-selection experiment (Section 5 of the paper).

For each saved evaluation matrix ``simu_n={n}_m={m}_random{idx}.pt``,
constructs a value function (linear / concave / convex / concave&convex),
derives pairwise preferences over alternatives, and optionally injects
cognitive bias (preference reversals on the smallest-margin pairs).

Saves the resulting reference info as JSON alongside the input ``.pt`` files
in ``data/simulated/``.

The reproduction package already ships the generated ``.json`` files; this
script is only needed to regenerate the simulated dataset from scratch.
"""

import sys
import os

import torch

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----

sys.path.append(f"{CODE_ROOT}/data_generation")
from reference_model import Reference, set_seed


if __name__ == "__main__":
    # The 8 problem settings used in Section 5 of the paper:
    #   vary n with m=3:   (6, 3), (7, 3), (8, 3), (9, 3), (10, 3)
    #   vary m with n=8:   (8, 2), (8, 4), (8, 5)        [(8, 3) already above]
    PROBLEM_SIZES = [
        (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
        (8, 2), (8, 4), (8, 5),
    ]
    N_INSTANCES = 20  # 20 random instances per setting

    out_dir = f"{CODE_ROOT}/data/simulated"
    os.makedirs(out_dir, exist_ok=True)

    for n, m in PROBLEM_SIZES:
        for random_index in range(N_INSTANCES):
            set_seed(random_index)
            eval_matrix = torch.load(
                f"{out_dir}/simu_n={n}_m={m}_random{random_index}.pt"
            ).numpy()
            reference = Reference(
                eval_matrix,
                value_function_type='concave&convex',
                preferences_num=eval_matrix.shape[0] * (eval_matrix.shape[0] - 1) // 2,
                cognitive_biases=0,
            )
            reference.reference_save(
                f"{out_dir}/simu_n={n}_m={m}_random{random_index}.json"
            )
            print(f"[saved] n={n}, m={m}, random={random_index}")
