"""Sample u from the feasible region of SOR preference constraints.

For the degenerate m=2, intervals_num=1 case, closed-form uniform sampling
on a 1D interval is used. Otherwise, the R package ``volesti`` hit-and-run
sampler is invoked via rpy2.

``u_sampler``                     : take first feasible subset, return (N, m) tensor.
``u_sampler_SOR_for_time_test``   : take all subsets (<=5), additionally return
                                    inconsistency metrics + timing (for `fig:computation time`).
"""

import sys

import numpy as np
import torch
import rpy2.robjects as robjects


# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
)
# ---- end CODE_ROOT snippet ----

sys.path.append(f'{CODE_ROOT}/core')
from Constrain import constr_generate, constr_generate_for_time_test


# Try a decreasing (with sign flip) slack schedule; stop at the first value
# that yields a sampleable polytope.
R_SAMPLER_CODE = """
sample_from_feasible_region <- function(constr, epsilon) {
  try(
    {
        library(volesti)
        library(utils)
        library(Rcpp)

        data_matrix <- as.matrix(constr)
        num_rows <- nrow(data_matrix)
        num_cols <- ncol(data_matrix)

        if (num_rows == num_cols+1){
            data_matrix <- data_matrix[-num_cols, ]
            A <- data_matrix[, 1:num_cols-1]
            b <- data_matrix[, num_cols]

            P = Hpolytope(A = A, b = b)
            points =  sample_points(P, n = 10000, random_walk = list("walk" = "RDHR"))
            complete_points <- rbind(points, 1 - colSums(points))
        }else{
            data_matrix <- data_matrix[-num_cols, ]
            A <- data_matrix[, 1:num_cols-1]
            A[(num_cols):(num_rows-2), ] <- A[(num_cols):(num_rows-2), ] - data_matrix[(num_cols):(num_rows-2), num_cols]
            b <- -data_matrix[, num_cols]
            b[(num_cols):(num_rows-2)] <- b[(num_cols):(num_rows-2)] - epsilon
            b[num_rows-1] <- 1
            P = Hpolytope(A = A, b = b)
            points =  sample_points(P, n = 10000, random_walk = list("walk" = "RDHR"))
            complete_points <- rbind(points, 1 - colSums(points))
        }
        return(complete_points)
    },
    silent = TRUE
  )
}

sampler <- function(constr){
    epsilon <- c(10e-5, 10e-4, 10e-6, 10e-7, 10e-3, 10e-9, 10e-8, 10e-2, 10e-1, 0,
                 -10e-9, -10e-7, -10e-5, -10e-4, -10e-3,
                 10e-10, 10e-11, 10e-12, 10e-13, 10e-14, 10e-15, 10e-16, -10e-2)
    for (e in epsilon) {
        result <- sample_from_feasible_region(constr, e)
        if (!inherits(result, "try-error")) break
    }
    return(result)
}
"""


def _sample_m2_closed_form(constr, device):
    """Closed-form uniform sampling for the m=2, intervals_num=1 polytope
    (a 1D interval in u_1)."""
    constr = np.array(constr)
    preference_rows = constr[2:-1]
    u1_lb_set, u1_ub_set = [0], [1]
    for row in preference_rows:
        if row[0] - row[1] > 0:
            u1_ub_set.append(-row[1] / (row[0] - row[1]))
        elif row[0] - row[1] < 0:
            u1_lb_set.append(-row[1] / (row[0] - row[1]))
    u1_lb, u1_ub = max(u1_lb_set), min(u1_ub_set)

    u1 = np.random.uniform(u1_lb, u1_ub, 10000)
    u_samples = np.stack([u1, 1 - u1], axis=-1)
    return torch.tensor(u_samples, dtype=torch.float32).to(device)


def _sample_hitandrun(constr, device):
    """Call R hit-and-run (volesti) on a general m >= 2 polytope."""
    r_matrix = robjects.r['matrix'](
        robjects.FloatVector(sum(constr, [])),
        nrow=len(constr), byrow=True,
    )
    robjects.r(R_SAMPLER_CODE)
    samples = np.array(robjects.globalenv['sampler'](r_matrix))
    return torch.tensor(samples, dtype=torch.float32).to(device).T


def _sample_one(constr, device):
    """Dispatch to closed-form or hit-and-run depending on the constraint shape."""
    if np.array(constr).shape[1] == 2:
        return _sample_m2_closed_form(constr, device)
    return _sample_hitandrun(constr, device)


def u_sampler(reference, evalmatrix, intervals_num, device):
    """Sample u for the first feasible preference subset."""
    constr = constr_generate(reference, evalmatrix, intervals_num, device)[0]
    return _sample_one(constr, device)


def u_sampler_SOR_for_time_test(reference, evalmatrix, intervals_num, device):
    """Sample u for every (up-to-5) feasible preference subset. Also returns
    inconsistency metrics + LP wall-clock time used by ``run_timing.py``."""
    constr_set, min_incon_num, incon_check, incon_check_time = (
        constr_generate_for_time_test(reference, evalmatrix, intervals_num, device)
    )
    u_samples_set = [_sample_one(c, device) for c in constr_set[:5]]
    return u_samples_set, min_incon_num, incon_check, incon_check_time
