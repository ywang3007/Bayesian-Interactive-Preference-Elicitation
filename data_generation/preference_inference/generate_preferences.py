"""One-shot script to generate the DM preference-inference settings used by
the preference-inference experiment (Section 4 of the paper).

For each of the 8 real-world MCDA datasets (mmc2--mmc9) and every
combination of (value-function type, number of observed preferences,
cognitive-bias ratio, random seed), fit a synthetic value function and
derive observed pairwise preferences. Saves one JSON file per combination
to ``data/section4_settings/``.

The reproduction package does NOT ship the generated ``.json`` files
because the full sweep (~10k files) is large; run this script to
regenerate them. The analysis and draw scripts work entirely off the
shipped CSVs in ``results/section4/``, so most users do not need to run
this.
"""

import sys
import os

import pandas as pd

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----

sys.path.append(f"{CODE_ROOT}/data_generation")
from reference_model import Reference, set_seed


# Real-world MCDA datasets (mmc2 through mmc9). File names encode
# "a{alternatives}-g{criteria}" for quick reference; order matches the
# dataset indexing used throughout Section 4 of the paper.
DATASET_FILES = [
    "mmc2-a14-g5.xlsx",
    "mmc3-a31-g6.xlsx",
    "mmc4-a76-g6.xlsx",
    "mmc5-a51-g7.xlsx",
    "mmc6-a93-g4.xlsx",
    "mmc7-a70-g7.xlsx",
    "mmc8-a86-g8.xlsx",
    "mmc9-a118-g8.xlsx",
]

VALUE_FUNCTION_TYPES = ['linear', 'concave', 'convex', 'concave&convex']
PREFERENCES_NUMS = [10, 20, 30, 40]       # number of observed pairwise comparisons (Table "factors" in §4.1)
COGNITIVE_BIASES = [0, 0.1, 0.2, 0.3]     # bias ratio
N_INSTANCES = 20                           # 20 random instances per combination


if __name__ == "__main__":
    out_dir = f"{CODE_ROOT}/data/section4_settings"
    os.makedirs(out_dir, exist_ok=True)

    for data_set_index, fname in enumerate(DATASET_FILES):
        data_path = f"{CODE_ROOT}/data/real_world/{fname}"
        data = pd.read_excel(data_path, sheet_name=0, engine='openpyxl').iloc[1:, 1:].to_numpy()

        for vft in VALUE_FUNCTION_TYPES:
            for pref_num in PREFERENCES_NUMS:
                for bias in COGNITIVE_BIASES:
                    for random_index in range(N_INSTANCES):
                        out_path = (f"{out_dir}/settings_dataset{data_set_index}"
                                    f"_{vft}_{pref_num}_{bias:f}_random{random_index}.json")
                        if os.path.exists(out_path):
                            continue
                        set_seed(random_index)
                        reference = Reference(data, vft, pref_num, bias)
                        reference.reference_save(out_path)
                        print(f"[saved] dataset{data_set_index}_{vft}_{pref_num}_{bias}_random{random_index}")
