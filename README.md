# Reproduction Package — IISE Transactions UIIE-9008

> *A Bayesian Interactive Preference Elicitation Framework Based on
> Monte Carlo Tree Search for Multiple Criteria Decision Aiding.*

This package regenerates every figure and table in the paper (main text +
supplementary) from the shipped CSVs, and supports re-running every
experiment from scratch.

---

## Quickstart — regenerate all paper figures/tables

All pre-computed CSVs are shipped under `results/`. The following commands
read those CSVs and write figures into `figures_generated/`.

```powershell
# §4 — preference inference (real datasets)
cd experiments/section4_preference_inference
python plot_ASP.py                           # Fig. 5-7   (§4)
python summarize_ASP.py                      # Tab. 4     (§4) + Tab. E.5 / E.6 (Wilcoxon)
python plot_time.py                          # Fig. 8     (§4.3)
python summarize_infeasible.py               # Tab. F.8   (SOR infeasibility stats)

# §5 — questioning policy (simulated instances)
cd ../section5_questioning_policy
python plot_main.py                          # Fig. 9-11  (§5.2)
python plot_wilcoxon.py                      # Tab. E.7   (prints LaTeX rows to stdout)
python aggregate_timing.py --per_instance    # Tab. 5     (§5.3)
python plot_sensitivity.py                   # Fig. H.3   (App H) + Fig. I.4/I.5 (App I)
python plot_noise.py                         # Fig. J.6   (App J)
```

Main-text uses sequential Fig./Tab. numbers (1–11 figs, 1–5 tabs).
Supplementary uses `<Letter>.<N>` where the letter is the appendix section
and `N` is a counter continuous across the whole supplement (e.g., Tab. E.7
is the 7th table of the supplement, in Appendix E).

---

## Layout

```
code/
├── data/                              inputs
│   ├── real_world/                    8 MCDA datasets (mmc2–mmc9.xlsx)
│   ├── simulated/                     synthetic instances for §5 (JSON)
│   ├── section4_settings/             preference settings for §4 (not shipped, see Notes)
│   └── best_intervals_num.json        γ (sub-interval count) per §5 instance
│
├── core/                              shared library
│   ├── SGVI.py                        Dirichlet VI + Bradley-Terry likelihood
│   ├── Measures.py                    fVAR / fPWI / fRAI / entropy
│   ├── Constrain.py, U_Sampler.py     SOR feasible-region + hit-and-run (Gurobi + R)
│   └── NpEncoder.py
│
├── data_generation/                   scripts that produced data/simulated/ and data/section4_settings/
│
├── experiments/
│   ├── section4_preference_inference/
│   │   ├── run_inference.py           SGVI + DIC γ-selection
│   │   ├── run_timing.py              SGVI vs. SOR wall-clock
│   │   ├── compute_ASP.py             .pt → ASP_results.csv
│   │   ├── summarize_ASP.py           + Wilcoxon table
│   │   ├── plot_ASP.py, plot_time.py
│   │
│   └── section5_questioning_policy/
│       ├── MCTS.py                    unified MCTS runner (Var / Ent, any Cp, any F3)
│       ├── H_O_X_d.py, H_DVF.py, H_RAND.py   heuristic baselines
│       ├── launch_noise_experiments.py       GPU+CPU scheduler
│       ├── noise_utils.py                    cognitive-bias injection helper
│       ├── evaluate_main.py                  sequences → results.csv
│       ├── evaluate_CB.py                    App H CSV
│       ├── evaluate_noise.py                 App J CSV
│       ├── aggregate_timing.py               per-instance decision-time summary
│       └── plot_{main,sensitivity,noise,wilcoxon}.py
│
├── results/                           shipped CSVs (sufficient to regenerate all figures)
└── figures_generated/                 reference figures for diff/compare
```

---

## Setup

```bash
conda create -n uiie9008 python=3.8 -c conda-forge -y
conda activate uiie9008
conda install -c conda-forge rpy2 r-base r-rcpp r-rcppeigen r-bh -y
pip install -r requirements.txt gurobipy
R -e "install.packages('volesti', repos='https://cloud.r-project.org')"
```

Gurobi's free embedded license handles our MILPs (≤20 vars).

**`CODE_ROOT`** — every script resolves paths via this variable; leave
unset and each script falls back to inferring it from its own location.

Conceptual illustrations — Fig. 1–4 (process / MDP tree / MCTS schematic /
framework) and Fig. B.1–B.2 (piecewise-linear value function, Dirichlet
aggregation) — are hand-drawn and have no corresponding script.

---

## Full re-run from scratch

### §4 (real datasets, ~25k SGVI runs)

```powershell
cd experiments/section4_preference_inference
python run_inference.py       # → results/section4_sgvi_outputs/*.pt
python compute_ASP.py         # → results/section4/ASP_results.csv
python summarize_ASP.py       # → results/section4/ASP_table.csv + Wilcoxon
python run_timing.py          # → results/section4/time_results.csv
```

### §5 (simulated, 640 interactive processes for main experiment)

```powershell
cd experiments/section5_questioning_policy

# One instance (sanity check)
python MCTS.py --method var --random_index 0            # defaults: n=8, m=3, T=8, cb=300, cp=1/sqrt(2)

# Full main sweep + Appendix I (Cp = 0.2 / 0.5 / 1.0 / 1/sqrt(2))
python launch_noise_experiments.py --F3s 0.0 --random_indices 0..19
# (repeat with --mcts_extra --cp 0.2 / 0.5 / 1.0 for Appendix I)

# Noise sweep (Appendix J)
python launch_noise_experiments.py --F3s 0.1 0.2 0.3 --random_indices 0..19

# Evaluate sequences → CSVs
python evaluate_main.py       # → results/section5/results.csv
python evaluate_CB.py         # → results/section5/results_CBtest_MCTS_{Var,Ent}.csv
python evaluate_noise.py      # → results/section5/eval_results_noise.csv
```

Output locations picked by `MCTS.py` based on `F3`:
- `F3 == 0` → `results/section5_sequences/MCTS_result_simudata_[Entropy_]Cp{cp}/...`
- `F3 > 0`  → `results/section5/noise/sequences/...`

### Decision-time measurement (App §5.3)

```powershell
# Serial timing (one worker to avoid GPU/CPU contention)
python launch_noise_experiments.py --gpus 0 --mcts_workers_per_gpu 1 --heuristic_workers 1 \
    --F3s 0.0 --random_indices 0..9 \
    --results_root ../../results/section5 \
    --save_timing
python aggregate_timing.py --per_instance
```

---

## Notes

- Raw sequence `.pt` files are **not shipped** (large). The CSVs under
  `results/` are sufficient to regenerate every figure/table.
- `data/section4_settings/` (~68k JSON files) is also not shipped. §4
  plots and tables work entirely off the shipped `ASP_results.csv` /
  `time_results.csv`.
- SGVI hyperparameters in all §4 and §5 code match the values disclosed
  in §4.1 of the paper: `T=100, W_g=10000, δ=1e-5`, Adam
  (`η=0.01, β₁=0.9, β₂=0.999, ε=1e-7`).
- `fix_paths.py` is a one-off path-rewriting utility from the author's
  server tree to this release layout; already applied.
