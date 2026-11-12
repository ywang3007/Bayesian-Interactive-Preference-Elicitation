"""Run the full Section 4 preference-inference experiment.

For every combination of (dataset, value-function type, number of
preferences, bias ratio, random seed), fit the SGVI posterior and select
the optimal number of sub-intervals via the Deviance Information Criterion
(DIC). Results for each cell are cached under
``results/section4_sgvi_outputs/`` so the loop is restartable.

These raw outputs are then aggregated by ``analyze_ASP.py`` into
``results/section4/ASP_results.csv``.
"""

import sys
import os
import json
import time
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

sys.path.append(f'{CODE_ROOT}/core')
from SGVI import SGVI


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def choose_best_intervals_num(evalmatrix, preferences_observed, preferences_unobserved, e, T, W_g, device):
    """Select the number of sub-intervals via DIC.

    Fits an SGVI posterior for each candidate ``intervals_num`` in
    ``[1, 2, 3, 4, 5]`` and returns the one with the lowest DIC along with
    the full inference trajectories for each candidate.
    """
    best_intervals_num = 0
    best_DIC = float('inf')
    phi_set_all, grad_set_all, L_set_all, V_all, DIC_all = [], [], [], [], []

    for intervals_num in [1, 2, 3, 4, 5]:
        sgvi = SGVI(evalmatrix, intervals_num, preferences_observed,
                    e, T, W_g, device, temperature=80)
        phi_set, grad_set, L_set = sgvi.sga()

        phi_set_all.append(phi_set)
        grad_set_all.append(grad_set)
        L_set_all.append(L_set)
        V_all.append(sgvi.V)

        if torch.isnan(phi_set).any():
            nan_index = torch.nonzero(torch.isnan(phi_set).sum(dim=1))[0].item()
            phi = phi_set[nan_index - 1]
        else:
            phi = torch.mean(phi_set[-50:], dim=0)

        u_pos = (phi ** 2) / torch.sum(phi ** 2)
        u_samples = torch.distributions.dirichlet.Dirichlet(phi ** 2).sample((10000,))

        DIC = (-4 * torch.mean(sgvi.loglikelihood(u_samples, preferences_observed))
               + 2 * sgvi.loglikelihood(u_pos.unsqueeze(0), preferences_observed))
        DIC_all.append(DIC)

        if DIC < best_DIC:
            best_intervals_num = intervals_num
            best_DIC = DIC

    print(f'[best_intervals_num] {best_intervals_num}')
    return best_intervals_num, phi_set_all, grad_set_all, L_set_all, V_all, DIC_all


for data_set_index in range(8):
    for value_function_type in ['linear', 'concave', 'convex', 'concave&convex']:
        for preferences_num in [10, 20, 30, 40]:
            for cognitive_biases in [0, 0.1, 0.2, 0.3]:
                for random_index in range(20):
                    result_path = (f"{CODE_ROOT}/results/section4_sgvi_outputs/"
                                   f"SGVIresults_dataset{data_set_index}_{value_function_type}"
                                   f"_{preferences_num}_{cognitive_biases:f}_random{random_index}.pt")
                    if os.path.exists(result_path):
                        continue

                    print(f'=== dataset{data_set_index}_{value_function_type}'
                          f'_{preferences_num}_{cognitive_biases}_random{random_index} ===')

                    setting_path = (f"{CODE_ROOT}/data/section4_settings/"
                                    f"settings_dataset{data_set_index}_{value_function_type}"
                                    f"_{preferences_num}_{cognitive_biases:f}_random{random_index}.json")
                    with open(setting_path, 'r') as fp:
                        setting = json.load(fp)
                    evalmatrix = torch.tensor(setting['evalmatrix']).to(device)
                    preferences_observed = torch.tensor(setting['preferences_observed']).to(device)
                    preferences_all = torch.tensor(setting['preferences_all']).to(device)
                    preferences_unobserved = preferences_all[
                        ~torch.all(preferences_all == preferences_observed[:, None], dim=2).any(0)
                    ]

                    e = 0.00001
                    T = 100
                    W_g = 10000

                    t0 = time.time()
                    (best_intervals_num, phi_set_all, grad_set_all,
                     L_set_all, V_all, DIC_all) = choose_best_intervals_num(
                        evalmatrix, preferences_observed, preferences_unobserved,
                        e, T, W_g, device,
                    )
                    print(f'  instance runtime: {time.time() - t0:.1f}s')

                    result = {
                        'e': e,
                        'T': T,
                        'W_g': W_g,
                        'best_intervals_num': best_intervals_num,
                        'phi_set_all': phi_set_all,
                        'grad_set_all': grad_set_all,
                        'L_set_all': L_set_all,
                        'V_all': V_all,
                        'DIC_all': DIC_all,
                        'data_set_index': data_set_index,
                        'value_function_type': value_function_type,
                        'preferences_num': preferences_num,
                        'cognitive_biases': cognitive_biases,
                        'random_index': random_index,
                    }
                    os.makedirs(os.path.dirname(result_path), exist_ok=True)
                    torch.save(result, result_path)
