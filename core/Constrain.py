"""Build the half-space constraint matrix for the SOR feasible region.

``constr_generate`` solves an LP to detect preference-inconsistency; if any
is found, it enumerates up to 5 minimum-cardinality conflict sets and
returns one constraint matrix per feasible-after-removal preference subset.

``constr_generate_for_time_test`` is a timing-instrumented variant used by
``experiments/section4_preference_inference/run_timing.py`` -- it returns
additional metrics for the `fig:computation time` figure.
"""

import sys
import math
import time

import numpy as np
import gurobipy as grb


# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
)
# ---- end CODE_ROOT snippet ----

sys.path.append(f'{CODE_ROOT}/core')
from SGVI import Disaggregation


MAX_INCON_SETS = 5          # cap on minimum-cardinality conflict sets to enumerate
BIG_M = 10
E_CONST = 1e-5              # strict-inequality slack
INCON_OBJECTIVE_SENSE = grb.GRB.MINIMIZE


def _resolve_feasibility(preferences_observed, V, u_num, y_num):
    """Return (preferences_subset_list, min_incon_num, incon_check).

    If the original preference set is feasible, returns it unchanged.
    Otherwise finds up to ``MAX_INCON_SETS`` minimum-cardinality subsets
    whose removal makes the system feasible.
    """
    # Feasibility LP: maximize the strict-inequality slack e
    m = grb.Model('feasibility')
    u = m.addMVar(u_num, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
    e = m.addVar(lb=-100, ub=grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS)
    value = V.T @ u
    for i in range(y_num):
        m.addConstr(value[preferences_observed[i][0]] >= value[preferences_observed[i][1]] + e)
    m.addConstr(u.sum() == 1)
    m.setObjective(e, sense=grb.GRB.MAXIMIZE)
    m.optimize()

    if m.objVal > 0:
        return [preferences_observed], 0, 0

    # Infeasible: find minimum-cardinality conflict sets using big-M MILP
    incon_set = []
    m = grb.Model('bigM')
    u = m.addMVar(u_num, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
    y = m.addVars(y_num, lb=0, ub=1, vtype=grb.GRB.INTEGER)
    value = V.T @ u
    for i in range(y_num):
        m.addConstr(value[preferences_observed[i][0]] + BIG_M * y[i]
                    >= value[preferences_observed[i][1]] + E_CONST)
    m.addConstr(u.sum() == 1)
    m.setObjective(y.sum(), sense=INCON_OBJECTIVE_SENSE)
    m.optimize()
    obj_Val = m.objVal

    # First minimum-cardinality conflict
    y_vals = [math.ceil(m.getVars()[u_num + i].x) for i in range(y_num)]
    incon = [i for i, x in enumerate(y_vals) if x == 1]
    min_incon_num = len(incon)
    incon_set.append(incon)

    # Enumerate additional conflicts of the same minimum size
    m.addConstr(sum(y[i] for i in incon) <= obj_Val - 1)
    while True:
        m.setObjective(y.sum(), sense=INCON_OBJECTIVE_SENSE)
        m.optimize()
        if m.status == grb.GRB.INFEASIBLE:
            break
        y_vals = [math.ceil(m.getVars()[u_num + i].x) for i in range(y_num)]
        incon = [i for i, x in enumerate(y_vals) if x == 1]
        if len(incon) != min_incon_num:
            break
        incon_set.append(incon)
        m.addConstr(sum(y[i] for i in incon) <= m.objVal - 1)

    incon_set = incon_set[:MAX_INCON_SETS]
    subsets = [[row for i, row in enumerate(preferences_observed) if i not in incon]
               for incon in incon_set]
    return subsets, min_incon_num, 1


def _build_constr_matrices(V, u_num, preferences_subsets):
    """For each preference subset, build a half-space constraint matrix:
       nonneg (K rows) + preference (|P| rows) + sum-to-one (1 row)."""
    constr_set = []
    for pref_subset in preferences_subsets:
        nonneg = [[-1 if x == y else 0 for x in range(u_num)] for y in range(u_num)]
        preference_rows = [(V.T[p[1]] - V.T[p[0]]).tolist() for p in pref_subset]
        sum_to_one = [[1] * u_num]
        constr_set.append(nonneg + preference_rows + sum_to_one)
    return constr_set[:MAX_INCON_SETS]


def _constr_generate_core(preferences_observed, evalmatrix, intervals_num, device):
    """Shared core: returns (constr_set, min_incon_num, incon_check)."""
    disaggregation = Disaggregation(evalmatrix, intervals_num, device)
    V = np.array(disaggregation.V.cpu())
    u_num = disaggregation.K
    y_num = len(preferences_observed)

    if y_num == 0:
        # No preferences: only nonneg + sum-to-one constraints
        nonneg = [[-1 if x == y else 0 for x in range(u_num)] for y in range(u_num)]
        return [nonneg + [[1] * u_num]], 0, 0

    pref_subsets, min_incon_num, incon_check = _resolve_feasibility(
        preferences_observed, V, u_num, y_num,
    )
    constr_set = _build_constr_matrices(V, u_num, pref_subsets)
    return constr_set, min_incon_num, incon_check


def constr_generate(preferences_observed, evalmatrix, intervals_num, device):
    """Return a list of constraint matrices (up to 5) covering all
    minimum-cardinality feasible subsets of the observed preferences."""
    constr_set, _, _ = _constr_generate_core(
        preferences_observed, evalmatrix, intervals_num, device,
    )
    return constr_set


def constr_generate_for_time_test(preferences_observed, evalmatrix, intervals_num, device):
    """Same as ``constr_generate`` but also returns inconsistency metrics and
    wall-clock time of the feasibility/inconsistency-resolution LP phase."""
    t0 = time.time()
    constr_set, min_incon_num, incon_check = _constr_generate_core(
        preferences_observed, evalmatrix, intervals_num, device,
    )
    return constr_set, min_incon_num, incon_check, time.time() - t0
