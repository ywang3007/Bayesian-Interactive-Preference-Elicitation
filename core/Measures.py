"""Uncertainty measures and goodness-of-fit scores used in Sections 4 and 5."""

import numpy as np
import torch
import gurobipy as grb


def var_calculate(phi_set_conv):
    """Average marginal variance of a Dirichlet(phi**2) over the K components."""
    theta = phi_set_conv ** 2
    theta_0 = torch.sum(theta, dim=1).view(-1, 1)
    var = theta * (theta_0 - theta) / (theta_0 ** 2 * (theta_0 + 1))
    return torch.mean(var)


def entropy_calculate(u_samples, V, all_action_set):
    """Mean binary entropy of pairwise-preference probabilities under samples u."""
    value = u_samples @ V
    v1 = value[:, all_action_set[:, 0]]
    v2 = value[:, all_action_set[:, 1]]
    p1 = torch.gt(v1, v2).sum(dim=0) / u_samples.shape[0]
    p2 = 1 - p1
    eps = 1e-12
    entropy = -(p1 * torch.log(p1 + eps) + p2 * torch.log(p2 + eps))
    return torch.mean(entropy)


def fPWI_calculate(u_samples, V, all_action_set):
    """Normalized pairwise-winning-index uncertainty (Section 5 metric)."""
    value = u_samples @ V
    v1 = value[:, all_action_set[:, 0]]
    v2 = value[:, all_action_set[:, 1]]
    p1 = torch.gt(v1, v2).sum(dim=0) / u_samples.shape[0]
    p2 = 1 - p1
    entropy = -p1 * torch.log(p1) - p2 * torch.log(p2)
    entropy = torch.where(torch.isnan(entropy), torch.zeros_like(entropy), entropy)
    n = V.shape[1]
    return torch.sum(entropy) / (n * (n - 1))


def fRAI_calculate(u_samples, V, all_action_set):
    """Normalized rank-acceptability-index entropy (Section 5 metric)."""
    sample_nums = u_samples.shape[0]
    value = u_samples @ V
    _, sorted_indices = torch.sort(value, dim=1, descending=True)
    ranked_indices = torch.argsort(sorted_indices, dim=1)
    n = V.shape[1]

    RAI = torch.zeros(n, n)
    for a in range(n):
        for k in range(n):
            RAI[a, k] = torch.bincount(ranked_indices[:, a], minlength=n)[k] / sample_nums

    fRAI = -RAI * torch.log2(RAI)
    fRAI = torch.where(torch.isnan(fRAI), torch.zeros_like(fRAI), fRAI)
    return torch.sum(fRAI) / n


def ASP_calculate(u_samples, V, preferences_all):
    """Average Share of Preserved pairwise orderings (Section 4 accuracy metric)."""
    n = V.shape[1]
    value = u_samples @ V
    v1 = value[:, preferences_all[:, 0]]
    v2 = value[:, preferences_all[:, 1]]
    POI = torch.ge(v1, v2).sum(dim=0) / u_samples.shape[0]
    return torch.sum(POI) * 2 / (n * (n - 1))


def fQUE_calculate(preferences_observed, V, all_action_set):
    """Count of pairs whose preference direction is NOT necessarily implied by
    the observed preferences (Section 5 look-ahead heuristic)."""
    V = np.array(V.cpu())
    u_num = V.shape[0]
    y_num = len(preferences_observed)

    # Feasibility check (does an e > 0 satisfy all preference constraints?)
    model = grb.Model('feasibility')
    u = model.addMVar(u_num, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
    e = model.addVar(lb=-100, ub=grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS)
    value = V.T @ u
    for i in range(y_num):
        model.addConstr(value[preferences_observed[i][0]] >= value[preferences_observed[i][1]] + e)
    model.addConstr(u.sum() == 1)
    model.setObjective(e, sense=grb.GRB.MAXIMIZE)
    model.optimize()

    f_NEC = 0
    e_const = 1e-6
    for relation_test in all_action_set:
        # Test if a > b is NOT implied (min(v_a - v_b) >= 0)
        for direction in [(0, 1), (1, 0)]:
            model = grb.Model('necessary_relation')
            u = model.addMVar(u_num, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
            value = V.T @ u
            for i in range(y_num):
                model.addConstr(value[preferences_observed[i][0]]
                                >= value[preferences_observed[i][1]] + e_const)
            model.addConstr(u.sum() == 1)
            model.setObjective(value[relation_test[direction[0]]]
                               - value[relation_test[direction[1]]],
                               sense=grb.GRB.MINIMIZE)
            model.optimize()
            if model.objVal >= 0:
                f_NEC += 1
                break

    n = V.shape[1]
    return n * (n - 1) / 2 - f_NEC


def NEC_test(preferences_observed, V, relation_test):
    """Return True iff preference ``relation_test`` is necessarily implied by
    ``preferences_observed`` (i.e., holds for every feasible u)."""
    V = np.array(V.cpu())
    u_num = V.shape[0]
    y_num = len(preferences_observed)

    model = grb.Model('necessary_relation')
    u = model.addMVar(u_num, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
    value = V.T @ u
    e_const = 1e-6
    for i in range(y_num):
        model.addConstr(value[preferences_observed[i][0]]
                        >= value[preferences_observed[i][1]] + e_const)
    model.addConstr(u.sum() == 1)
    model.setObjective(value[relation_test[0]] - value[relation_test[1]],
                       sense=grb.GRB.MINIMIZE)
    model.optimize()
    return model.objVal >= 0
