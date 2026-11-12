"""
Cognitive-bias noise injection shared by all methods in the noise experiment.

The rule mirrors ``reference_generate_formal.py`` (Section 5.1):
    1. For every non-dominated pair, compute |U_true(winner) - U_true(loser)|.
    2. Sort pairs by this difference in ascending order (closest preferences first).
    3. Flip the top ``ceil(K * cognitive_biases)`` pairs: winner <-> loser.

The flip is deterministic given (preferences_non_domi, value, cognitive_biases),
so both the method's online question selection and the final evaluation see the
same biased DM.
"""

import json
import math
import os

import torch

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


# Resolve the project root once; downstream scripts and CLI defaults use this.


def apply_cognitive_bias(preferences_non_domi, value, cognitive_biases):
    """Return a biased copy of ``preferences_non_domi``.

    Args:
        preferences_non_domi: LongTensor of shape (K, 2); each row is
            ``(winner_idx, loser_idx)`` for a non-dominated pair.
        value: 1D tensor or list of length ``n``; ``U_true`` for each
            alternative.
        cognitive_biases: float in [0, 1]; proportion of pairs to flip.

    Returns:
        LongTensor of shape (K, 2) with ``ceil(K * cognitive_biases)``
        rows swapped (winner <-> loser).
    """
    biased = preferences_non_domi.clone()
    if cognitive_biases <= 0:
        return biased

    device = biased.device
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32)
    value = value.to(device=device, dtype=torch.float32)

    winners = biased[:, 0].long()
    losers = biased[:, 1].long()
    diffs = value[winners] - value[losers]  # positive by construction

    k = biased.shape[0]
    n_flip = math.ceil(k * cognitive_biases)
    if n_flip <= 0:
        return biased

    _, flip_idx = torch.topk(diffs, n_flip, largest=False)
    swapped = biased[flip_idx].clone()
    biased[flip_idx] = swapped[:, [1, 0]]
    return biased


def load_setting(setting_path, device):
    """Load an instance setting and return key tensors + the true U vector."""
    with open(setting_path, "r") as fp:
        setting = json.load(fp)
    evalmatrix = torch.tensor(setting["evalmatrix"]).to(device)
    preferences_all = torch.tensor(setting["preferences_all"]).to(device)
    preferences_non_domi = torch.tensor(setting["preferences_non_domi"]).to(device)
    preferences_non_domi_index = torch.tensor(setting["preferences_non_domi_index"]).to(device)
    value = torch.tensor(setting["value"], dtype=torch.float32).to(device)
    return (
        evalmatrix,
        preferences_all,
        preferences_non_domi,
        preferences_non_domi_index,
        value,
    )


def biased_dm_from_setting(setting_path, cognitive_biases, device):
    """Convenience wrapper that returns the biased DM_answer tensor and
    all other objects a method typically needs."""
    (
        evalmatrix,
        preferences_all,
        preferences_non_domi,
        preferences_non_domi_index,
        value,
    ) = load_setting(setting_path, device)
    DM_answer = apply_cognitive_bias(preferences_non_domi, value, cognitive_biases)
    return (
        evalmatrix,
        preferences_all,
        preferences_non_domi,
        preferences_non_domi_index,
        value,
        DM_answer,
    )


def sequence_output_path(results_root, method, n, m, T, random_index, cognitive_biases):
    """Canonical output path for a question sequence."""
    f3_str = f"{cognitive_biases:.2f}"
    fname = (
        f"{method}_n{n}_m{m}_T{T}_random{random_index}_F3{f3_str}.pt"
    )
    out_dir = os.path.join(results_root, "sequences")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, fname)


def timing_output_path(results_root, method, n, m, T, random_index, cognitive_biases):
    """Canonical output path for per-round decision timing (list of floats).
    Only produced when a runner is launched with ``--save_timing``.
    """
    f3_str = f"{cognitive_biases:.2f}"
    fname = (
        f"{method}_n{n}_m{m}_T{T}_random{random_index}_F3{f3_str}_timing.pt"
    )
    out_dir = os.path.join(results_root, "timings")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, fname)
