"""The DM's preference model, used for synthesising pairwise preferences.

Shared by both data-generation pipelines:
  - ``question_selection/generate_preferences.py`` (synthetic instances)
  - ``preference_inference/generate_preferences.py`` (real-world instances)

The value-function construction follows Ghaderi & Kadzinski (2021).
"""

import sys
import os
import json
import math
import random

import numpy as np

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
)
# ---- end CODE_ROOT snippet ----

sys.path.append(f"{CODE_ROOT}/core")
from NpEncoder import NpEncoder


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Reference:
    """DM preference model + derived pairwise preferences for one instance.

    Parameters
    ----------
    evalmatrix : np.ndarray (n, m)
        Performance matrix.
    value_function_type : {'linear', 'concave', 'convex', 'concave&convex'}
        Family of marginal value functions.
    preferences_num : int
        Number of pairwise comparisons to sample from the non-dominated pool.
    cognitive_biases : float in [0, 1]
        Fraction of observed preferences to reverse (smallest-margin first)
        to simulate cognitive bias.
    """

    def __init__(self, evalmatrix, value_function_type, preferences_num, cognitive_biases):
        self.evalmatrix = evalmatrix.astype(float)
        self.n = evalmatrix.shape[0]
        self.m = evalmatrix.shape[1]
        self.value_function_type = value_function_type
        self.preferences_num = preferences_num
        self.cognitive_biases = cognitive_biases

        self.weight = self._get_weight()
        self.value, self.c = self._get_value()

        self.preferences_all = self._preferences_all_generate().astype(int)
        self.preferences_non_domi, self.preferences_non_domi_index = self._preferences_non_domi_generate()
        self.preferences_observed = self._preferences_observed_generate()

    @staticmethod
    def _f(x, c):
        return (1 - np.exp(-c * x)) / (1 - np.exp(-c))

    def _get_weight(self):
        return np.random.dirichlet(alpha=np.ones(self.m))

    @staticmethod
    def _get_c(a, b, m):
        return np.random.uniform(a, b, m)

    def _get_value(self):
        X = self.evalmatrix.copy()
        cri_max = np.max(X, axis=0)
        cri_min = np.min(X, axis=0)
        for i in range(self.m):
            X[:, i] = (X[:, i] - cri_min[i]) / (cri_max[i] - cri_min[i])

        if self.value_function_type == 'linear':
            return X @ self.weight, None
        if self.value_function_type == 'concave':
            c = self._get_c(0, 10, self.m)
            return self._f(X, c) @ self.weight, c
        if self.value_function_type == 'convex':
            c = self._get_c(-10, 0, self.m)
            return self._f(X, c) @ self.weight, c
        if self.value_function_type == 'concave&convex':
            c = self._get_c(-10, 10, self.m)
            return self._f(X, c) @ self.weight, c
        raise ValueError(f"unknown value_function_type: {self.value_function_type}")

    def _preferences_all_generate(self):
        """Full list of pairwise preferences (winner, loser) over all ``C(n,2)`` pairs."""
        preferences_num = self.n * (self.n - 1) // 2
        preferences = np.zeros((preferences_num, 2))
        k = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.value[i] > self.value[j]:
                    preferences[k][0], preferences[k][1] = i, j
                else:
                    preferences[k][0], preferences[k][1] = j, i
                k += 1
        return preferences

    def _preferences_non_domi_generate(self):
        """Restrict to pairs that are NOT Pareto-dominated in either direction."""
        dominated = []
        for i in range(self.preferences_all.shape[0]):
            a = self.evalmatrix[self.preferences_all[i][0]]
            b = self.evalmatrix[self.preferences_all[i][1]]
            if np.all(a >= b) or np.all(b >= a):
                dominated.append(i)
        non_domi_index = list(set(range(self.preferences_all.shape[0])) - set(dominated))
        return self.preferences_all[non_domi_index], non_domi_index

    def _preferences_observed_generate(self):
        """Uniformly sample ``preferences_num`` non-dominated pairs; then flip
        ``ceil(preferences_num * cognitive_biases)`` of them (smallest-margin first)."""
        sampled_index = random.sample(range(self.preferences_non_domi.shape[0]), self.preferences_num)
        observed = self.preferences_non_domi[sampled_index]
        if self.cognitive_biases > 0:
            biases_num = math.ceil(self.preferences_num * self.cognitive_biases)
            difference = self.value[observed[:, 0]] - self.value[observed[:, 1]]
            biases_index = np.argsort(difference)[:biases_num]
            for index in biases_index:
                pair = observed[index].copy()
                observed[index][0], observed[index][1] = pair[1], pair[0]
        return observed

    def reference_save(self, path):
        reference = {
            'evalmatrix': self.evalmatrix,
            'value_function_type': self.value_function_type,
            'preferences_num': self.preferences_num,
            'cognitive_biases': self.cognitive_biases,
            'weight': self.weight,
            'c': self.c,
            'value': self.value,
            'preferences_all': self.preferences_all,
            'preferences_non_domi': self.preferences_non_domi,
            'preferences_non_domi_index': self.preferences_non_domi_index,
            'preferences_observed': self.preferences_observed,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(reference, f, indent=4, sort_keys=False, cls=NpEncoder)
