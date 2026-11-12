"""
Unified MCTS runner for Section 5 (main) + Appendix H / I / J.

Defaults match the main-paper configuration: ``--cb 300`` (computation
budget) and ``--cp 1/sqrt(2)`` (UCT exploration constant).

Typical single-instance invocations
-----------------------------------

    # Main-paper MCTS_Var / MCTS_Ent (uses defaults cb=300, cp=1/sqrt(2))
    python MCTS.py --method var --n 8 --m 3 --T 8 --random_index 0 --gpu 0
    python MCTS.py --method ent --n 8 --m 3 --T 8 --random_index 0 --gpu 0

    # Appendix H  (computation-budget sensitivity): sweep --cb over {100,200,300,400,500}
    python MCTS.py --method var --cb 100 --n 8 --m 3 --T 8 --random_index 0 --gpu 0

    # Appendix I  (C_p exploration-parameter sensitivity): sweep --cp over {0.2, 0.5, 1.0, 1/sqrt(2)}
    python MCTS.py --method var --cp 0.5 --n 8 --m 3 --T 8 --random_index 0 --gpu 0

    # Appendix J  (noise robustness); --F3>0 switches output layout
    python MCTS.py --method ent --F3 0.2 --n 8 --m 3 --T 8 --random_index 0 --gpu 0

Full-sweep examples (bash)
--------------------------

Appendix H: 2 methods x 5 CB values x 8 problem settings x 20 instances
::

    for method in var ent; do
      for cb in 100 200 300 400 500; do
        for nm in "6 3" "7 3" "8 3" "9 3" "10 3" "8 2" "8 4" "8 5"; do
          set -- $nm
          for idx in $(seq 0 19); do
            python MCTS.py --method $method --cb $cb --n $1 --m $2 --T 8 \\
                --random_index $idx --gpu 0
          done
        done
      done
    done
    # Then: python evaluate_CB.py && python plot_sensitivity.py

Appendix I: 2 methods x 4 Cp values x 8 problem settings x 20 instances
::

    for method in var ent; do
      for cp in 0.2 0.5 0.7071067811865475 1.0; do
        for nm in "6 3" "7 3" "8 3" "9 3" "10 3" "8 2" "8 4" "8 5"; do
          set -- $nm
          for idx in $(seq 0 19); do
            python MCTS.py --method $method --cp $cp --n $1 --m $2 --T 8 \\
                --random_index $idx --gpu 0
          done
        done
      done
    done
    # Then: python evaluate_main.py && python plot_sensitivity.py

Appendix J: all 8 heuristic policies with F3>0; use the dedicated scheduler
``launch_noise_experiments.py`` instead of looping manually.

Output layout
-------------

``F3 == 0`` (main paper, Appendix H, Appendix I):
    ``<results_root>/MCTS_result_simudata_[Entropy_]Cp{cp}/``
    ``new_node_sequence_n{n}_m{m}_random{idx}_MAX{T}_BUDGET{cb}.pt``
    (readable by ``evaluate_main.py`` and ``evaluate_CB.py``)

``F3 > 0`` (Appendix J):
    ``<results_root>/noise/sequences/``
    ``MCTS_{Var|Ent}_n{n}_m{m}_T{T}_random{idx}_F3{F3:.2f}.pt``
    (readable by ``evaluate_noise.py``)
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


# ---- CODE_ROOT (portable paths) ----
CODE_ROOT = os.environ.get(
    "CODE_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.append(os.path.join(CODE_ROOT, "core"))

from SGVI import SGVI, Disaggregation
from Measures import entropy_calculate, var_calculate


# ---- globals (set in main) ----
device = None
MAX_ROUND_NUMBER = None
COMPUTATION_BUDGET = None
C_p = None
REWARD_FN_NAME = None  # 'ent' or 'var'

evalmatrix = None
preferences_non_domi_index = None
DM_answer = None
all_action_set = None
action_num = None

# SGVI hyper-parameters (match Section 4.1 in the paper)
e = 0.00001        # convergence tolerance delta
T_sgvi = 100       # max Adam iterations
W_g = 10000        # MC samples for the stochastic gradient
conv_num = 50      # window used to average phi
samples_num = 10000  # posterior sample size (= W)
intervals_num = None

reward_cache = None
prefix_phi_cache = None
prior_phi_ones = None
disaggregation_V = None
empty_reference = None


# ---- cognitive-bias injection (F3 > 0) ----
def apply_cognitive_bias(preferences_non_domi, value, cognitive_biases):
    """Flip ``ceil(K * F3)`` closest pairs (by |U_true(winner) - U_true(loser)|)."""
    biased = preferences_non_domi.clone()
    if cognitive_biases <= 0:
        return biased

    dev = biased.device
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32)
    value = value.to(device=dev, dtype=torch.float32)

    winners = biased[:, 0].long()
    losers = biased[:, 1].long()
    diffs = value[winners] - value[losers]
    k = biased.shape[0]
    n_flip = math.ceil(k * cognitive_biases)
    if n_flip <= 0:
        return biased
    _, flip_idx = torch.topk(diffs, n_flip, largest=False)
    swapped = biased[flip_idx].clone()
    biased[flip_idx] = swapped[:, [1, 0]]
    return biased


# ---- MCTS state / node ----
class State:
    def __init__(self, candidate_choices):
        self.current_round_index = 0
        self.cumulative_choices = []
        self.candidate_choices = candidate_choices
        self.state_id = None

    def get_current_round_index(self):
        return self.current_round_index

    def set_current_round_index(self, turn):
        self.current_round_index = turn

    def get_cumulative_choices(self):
        return self.cumulative_choices

    def set_cumulative_choices(self, choices):
        self.cumulative_choices = choices
        self.state_id = "_".join(map(str, self.cumulative_choices))

    def get_state_id(self):
        return self.state_id

    def is_terminal(self):
        return self.current_round_index == MAX_ROUND_NUMBER

    def compute_reward(self, true_round, rollout_way):
        # Transposition detection: Bradley-Terry likelihood is permutation-
        # invariant over the observed pairwise comparisons, so two states
        # that reach the same {real preferences, rollout preferences} via
        # different orderings collapse to the same MDP state and share a
        # reward. Separating the "real" prefix from the rollout suffix
        # keeps rollout_way semantics (``random`` vs ``reference_based``)
        # distinct across the two frozensets.
        cache_key = (
            frozenset(self.cumulative_choices[:true_round]),
            frozenset(self.cumulative_choices[true_round:]),
            rollout_way,
        )
        if cache_key in reward_cache:
            return reward_cache[cache_key]

        if true_round == 0:
            phi = prior_phi_ones
            u = torch.distributions.dirichlet.Dirichlet(phi ** 2).sample((samples_num,))
            value_u = u @ disaggregation_V

            questions = all_action_set[self.cumulative_choices]
            reference_list = []
            for Q in questions:
                p = float((value_u[:, Q[0]] > value_u[:, Q[1]]).float().mean())
                reference_list.append(Q if p > 0.5 else Q[[1, 0]])

            reference = (
                torch.stack(reference_list, dim=0) if reference_list else empty_reference
            )
        else:
            true_choice_idx = self.cumulative_choices[:true_round]
            true_reference = DM_answer[true_choice_idx]

            if rollout_way == "random":
                future_choice_idx = self.cumulative_choices[true_round:]
                if future_choice_idx:
                    questions = all_action_set[future_choice_idx]
                    random_binary = torch.rand(questions.shape[0], device=device) > 0.5
                    reference = questions.clone()
                    if random_binary.any():
                        reference[random_binary] = questions[random_binary][:, [1, 0]]
                    reference = torch.cat((true_reference, reference), dim=0)
                else:
                    reference = true_reference
            elif rollout_way == "reference_based":
                # Transposition: SGVI posterior depends only on the SET of
                # observed preferences (BT likelihood is permutation-
                # invariant), so cache by frozenset — sibling branches that
                # share a prefix in different orderings now hit the cache.
                prefix_key = frozenset(true_choice_idx)
                if prefix_key in prefix_phi_cache:
                    phi_prefix, V_prefix = prefix_phi_cache[prefix_key]
                else:
                    sgvi_prefix = SGVI(
                        evalmatrix, intervals_num, true_reference, e, T_sgvi, W_g, device
                    )
                    phi_set_prefix, _, _ = sgvi_prefix.sga()

                    if not torch.isnan(phi_set_prefix).any():
                        phi_prefix_conv = phi_set_prefix[-conv_num:]
                    else:
                        nan_rows = torch.nonzero(
                            torch.isnan(phi_set_prefix).sum(dim=1), as_tuple=False
                        )
                        nan_index = nan_rows[0].item()
                        start_idx = max(0, nan_index - 10)
                        phi_prefix_conv = phi_set_prefix[start_idx:nan_index]

                    phi_prefix = torch.mean(phi_prefix_conv, dim=0)
                    V_prefix = sgvi_prefix.V
                    prefix_phi_cache[prefix_key] = (phi_prefix, V_prefix)

                u = torch.distributions.dirichlet.Dirichlet(phi_prefix ** 2).sample(
                    (samples_num,)
                )
                value_u = u @ V_prefix

                future_choice_idx = self.cumulative_choices[true_round:]
                reference_list = []
                if future_choice_idx:
                    questions = all_action_set[future_choice_idx]
                    for Q in questions:
                        p = float((value_u[:, Q[0]] > value_u[:, Q[1]]).float().mean())
                        reference_list.append(Q if p > 0.5 else Q[[1, 0]])
                if reference_list:
                    future_reference = torch.stack(reference_list, dim=0)
                    reference = torch.cat((true_reference, future_reference), dim=0)
                else:
                    reference = true_reference
            else:
                raise ValueError(f"Unknown rollout_way: {rollout_way}")

        sgvi = SGVI(evalmatrix, intervals_num, reference, e, T_sgvi, W_g, device)
        phi_set, _, _ = sgvi.sga()

        if not torch.isnan(phi_set).any():
            phi_set_conv = phi_set[-conv_num:]
        else:
            nan_rows = torch.nonzero(torch.isnan(phi_set).sum(dim=1), as_tuple=False)
            nan_index = nan_rows[0].item()
            start_idx = max(0, nan_index - 10)
            phi_set_conv = phi_set[start_idx:nan_index]

        if REWARD_FN_NAME == "ent":
            phi = torch.mean(phi_set_conv, dim=0)
            u_samples = torch.distributions.dirichlet.Dirichlet(phi ** 2).sample(
                (samples_num,)
            )
            reward = entropy_calculate(u_samples, sgvi.V, all_action_set)
        elif REWARD_FN_NAME == "var":
            reward = var_calculate(phi_set_conv)
        else:
            raise ValueError(f"Unknown REWARD_FN_NAME: {REWARD_FN_NAME}")

        final_reward = -reward
        reward_cache[cache_key] = final_reward
        return final_reward

    def get_next_state_with_random_choice(self):
        random_choice = random.choice(self.candidate_choices)
        next_state_candidate_choice = [x for x in self.candidate_choices if x != random_choice]
        next_state = State(next_state_candidate_choice)
        next_state.set_current_round_index(self.current_round_index + 1)
        next_state.set_cumulative_choices(self.cumulative_choices + [random_choice])
        return next_state


class Node:
    def __init__(self):
        self.parent = None
        self.children = []
        self.visit_times = 0
        self.quality_value = 0.0
        self.state = None

    def set_state(self, state): self.state = state
    def get_state(self): return self.state
    def set_parent(self, parent): self.parent = parent
    def get_children(self): return self.children
    def get_visit_times(self): return self.visit_times
    def visit_times_add_one(self): self.visit_times += 1
    def get_quality_value(self): return self.quality_value
    def quality_value_add_n(self, n): self.quality_value += n

    def is_all_expand(self):
        return len(self.children) == len(self.get_state().candidate_choices)

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)


def tree_policy(node):
    while not node.get_state().is_terminal():
        if node.is_all_expand():
            node = best_child(node, True)
        else:
            return expand(node)
    return node


def default_policy(node, true_round, rollout_way):
    current_state = node.get_state()
    while not current_state.is_terminal():
        current_state = current_state.get_next_state_with_random_choice()
    return current_state.compute_reward(true_round, rollout_way)


def expand(node):
    tried = [sub.get_state().get_state_id() for sub in node.get_children()]
    new_state = node.get_state().get_next_state_with_random_choice()
    while new_state.get_state_id() in tried:
        new_state = node.get_state().get_next_state_with_random_choice()
    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    return sub_node


def best_child(node, is_exploration):
    best_score = -sys.maxsize
    best_sub = None
    for sub in node.get_children():
        C = C_p if is_exploration else 0.0
        left = sub.get_quality_value() / sub.get_visit_times()
        right_in = 2.0 * math.log(node.get_visit_times()) / sub.get_visit_times()
        right = 2 * C * math.sqrt(right_in)
        score = left + right
        if score > best_score:
            best_sub = sub
            best_score = score
    return best_sub


def backup(node, reward):
    while node is not None:
        node.visit_times_add_one()
        node.quality_value_add_n(reward)
        node = node.parent


def monte_carlo_tree_search(node, true_round, rollout_way, progress_desc=""):
    pbar = tqdm(
        range(COMPUTATION_BUDGET),
        desc=progress_desc,
        leave=False,
        mininterval=1.0,
        ncols=80,
    )
    for _ in pbar:
        expand_node = tree_policy(node)
        reward = default_policy(expand_node, true_round, rollout_way)
        backup(expand_node, reward)
    pbar.close()
    return best_child(node, False)


def get_all_action_set(evalmatrix_local, preferences_non_domi_index_local):
    n = evalmatrix_local.shape[0]
    pair_num = int(n * (n - 1) / 2)
    all_pairs = torch.zeros((pair_num, 2), dtype=torch.int64).to(device)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs[k][0] = int(i)
            all_pairs[k][1] = int(j)
            k += 1
    return all_pairs[preferences_non_domi_index_local]


def run_one_instance(args, best_intervals_num_set):
    """Run a single (method, n, m, T, random_index, F3) MCTS instance."""
    global evalmatrix, preferences_non_domi_index, DM_answer, all_action_set
    global action_num, intervals_num
    global reward_cache, prefix_phi_cache
    global prior_phi_ones, disaggregation_V, empty_reference
    global MAX_ROUND_NUMBER

    MAX_ROUND_NUMBER = args.T

    setting_path = os.path.join(
        CODE_ROOT, "data", "simulated",
        f"simu_n={args.n}_m={args.m}_random{args.random_index}.json",
    )
    with open(setting_path, "r") as fp:
        setting = json.load(fp)
    evalmatrix = torch.tensor(setting["evalmatrix"]).to(device)
    preferences_non_domi = torch.tensor(setting["preferences_non_domi"]).to(device)
    preferences_non_domi_index = torch.tensor(setting["preferences_non_domi_index"]).to(device)
    value_true = torch.tensor(setting["value"], dtype=torch.float32).to(device)

    DM_answer = apply_cognitive_bias(preferences_non_domi, value_true, args.F3)

    all_action_set = get_all_action_set(evalmatrix, preferences_non_domi_index)
    action_num = all_action_set.shape[0]

    best_intervals_num = best_intervals_num_set[
        f"n{args.n}_m{args.m}_random{args.random_index}"
    ][0]
    intervals_num = best_intervals_num

    reward_cache = {}
    prefix_phi_cache = {}
    prior_phi_ones = torch.ones(evalmatrix.shape[1] * intervals_num, device=device)
    disaggregation = Disaggregation(evalmatrix, intervals_num, device)
    disaggregation_V = disaggregation.V
    empty_reference = torch.empty((0, 2), dtype=torch.int64, device=device)

    candidate_choices = list(range(action_num))
    init_state = State(candidate_choices)
    init_node = Node()
    init_node.set_state(init_state)
    current_node = init_node
    rollout_way = "reference_based"

    method_tag = "MCTS_Var" if args.method == "var" else "MCTS_Ent"
    instance_tag = (
        f"{method_tag} F3={args.F3:.2f} T={MAX_ROUND_NUMBER} rand={args.random_index}"
    )
    round_times = []
    for round_i in range(MAX_ROUND_NUMBER):
        desc = f"{instance_tag} | round {round_i + 1}/{MAX_ROUND_NUMBER}"
        t0_round = time.time()
        current_node = monte_carlo_tree_search(
            current_node, round_i, rollout_way, progress_desc=desc
        )
        round_times.append(time.time() - t0_round)

    node_sequence = [int(x) for x in current_node.get_state().get_cumulative_choices()]
    return node_sequence, round_times


def resolve_output_paths(args):
    """Decide (sequence_path, timing_path) based on F3 and method."""
    method_tag = "MCTS_Var" if args.method == "var" else "MCTS_Ent"

    if args.F3 == 0.0:
        # Legacy main-paper convention (matches server directory layout
        # read by evaluate_main.py).
        dir_prefix = "" if args.method == "var" else "Entropy_"
        subdir = f"MCTS_result_simudata_{dir_prefix}Cp{args.cp}"
        base = args.output_dir or os.path.join(
            CODE_ROOT, "results", "section5_sequences"
        )
        out_dir = os.path.join(base, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fname = (
            f"new_node_sequence_n{args.n}_m{args.m}_random{args.random_index}"
            f"_MAX{args.T}_BUDGET{args.cb}.pt"
        )
        sequence_path = os.path.join(out_dir, fname)
        timing_path = os.path.join(
            out_dir, fname.replace(".pt", "_timing.pt")
        )
    else:
        # Noise-experiment convention (read by evaluate_noise.py).
        base = args.output_dir or os.path.join(
            CODE_ROOT, "results", "section5", "noise"
        )
        seq_dir = os.path.join(base, "sequences")
        tim_dir = os.path.join(base, "timings")
        os.makedirs(seq_dir, exist_ok=True)
        os.makedirs(tim_dir, exist_ok=True)
        tag = (
            f"{method_tag}_n{args.n}_m{args.m}_T{args.T}"
            f"_random{args.random_index}_F3{args.F3:.2f}"
        )
        sequence_path = os.path.join(seq_dir, f"{tag}.pt")
        timing_path = os.path.join(tim_dir, f"{tag}_timing.pt")
    return sequence_path, timing_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--method", choices=["ent", "var"], required=True,
                   help="Reward function: 'var' (MCTS_Var) or 'ent' (MCTS_Ent)")
    p.add_argument("--n", type=int, default=8,
                   help="Number of alternatives (default 8, matching the "
                        "setting used in Appendices H/I/J)")
    p.add_argument("--m", type=int, default=3,
                   help="Number of criteria (default 3, matching the "
                        "setting used in Appendices H/I/J)")
    p.add_argument("--T", type=int, default=8,
                   help="Number of DM questions (default 8)")
    p.add_argument("--random_index", type=int, required=True,
                   help="Instance index (0..19 for main experiment)")
    p.add_argument("--F3", type=float, default=0.0,
                   help="Cognitive-bias fraction (0 = clean, 0.1/0.2/0.3 = Appendix J)")
    p.add_argument("--gpu", type=int, default=0,
                   help="Logical GPU index after CUDA_VISIBLE_DEVICES filtering")
    p.add_argument("--cb", type=int, default=300, help="MCTS computation budget")
    p.add_argument("--cp", type=float, default=1.0 / math.sqrt(2),
                   help="UCT exploration constant (default 1/sqrt(2))")
    p.add_argument("--output_dir", type=str, default="",
                   help="Override default output directory")
    p.add_argument("--best_intervals_json", type=str,
                   default=os.path.join(CODE_ROOT, "data",
                                        "best_intervals_num.json"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_timing", action="store_true",
                   help="Also save per-round wall-clock time")
    return p.parse_args()


def main():
    global device, COMPUTATION_BUDGET, C_p, REWARD_FN_NAME

    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    COMPUTATION_BUDGET = args.cb
    C_p = args.cp
    REWARD_FN_NAME = args.method

    sequence_path, timing_path = resolve_output_paths(args)
    if os.path.exists(sequence_path) and not args.overwrite:
        print(f"[skip] {sequence_path} already exists")
        return

    with open(args.best_intervals_json, "r") as fp:
        best_intervals_num_set = json.load(fp)

    t0 = time.time()
    node_sequence, round_times = run_one_instance(args, best_intervals_num_set)
    elapsed = time.time() - t0

    torch.save(node_sequence, sequence_path)
    if args.save_timing:
        torch.save(round_times, timing_path)

    method_tag = "MCTS_Var" if args.method == "var" else "MCTS_Ent"
    print(
        f"[done] method={method_tag} cp={args.cp:.4f} n={args.n} m={args.m} "
        f"T={args.T} random_index={args.random_index} F3={args.F3:.2f} "
        f"elapsed={elapsed:.1f}s -> {sequence_path}"
    )


if __name__ == "__main__":
    main()
