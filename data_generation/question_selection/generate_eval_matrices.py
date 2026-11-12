"""One-shot script to generate synthetic evaluation matrices used by the
question-selection experiment (Section 5 of the paper).

Samples an ``(n, m)`` performance matrix per random seed such that no row
Pareto-dominates another (full non-dominance), and saves each matrix to
``data/simulated/simu_n={n}_m={m}_random{idx}.pt``.

The reproduction package already ships the generated ``.pt`` files; this
script is only needed to regenerate the simulated dataset from scratch.
"""

import os
import random

import numpy as np
import torch


# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nondominance_relation_detect(eval_matrix):
    """Return True iff no row of ``eval_matrix`` Pareto-dominates another."""
    for i in range(eval_matrix.shape[0] - 1):
        for j in range(i + 1, eval_matrix.shape[0]):
            if torch.all(eval_matrix[i] <= eval_matrix[j]) or torch.all(eval_matrix[i] >= eval_matrix[j]):
                return False
    return True


def eval_matrix_generate(n, m):
    """Sample an ``(n, m)`` performance matrix with full non-dominance via
    rejection sampling: each new row is redrawn until it is non-dominated
    with every previously accepted row."""
    eval_matrix = torch.rand(n, m)
    for i in range(1, n):
        while True:
            new_row = torch.rand(m)
            if all(
                not (torch.all(new_row <= eval_matrix[j]) or torch.all(new_row >= eval_matrix[j]))
                for j in range(i)
            ):
                eval_matrix[i] = new_row
                break
    return eval_matrix


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
            eval_matrix = eval_matrix_generate(n, m)
            assert nondominance_relation_detect(eval_matrix)
            torch.save(
                eval_matrix,
                f"{out_dir}/simu_n={n}_m={m}_random{random_index}.pt",
            )
            print(f"[saved] n={n}, m={m}, random={random_index}")
