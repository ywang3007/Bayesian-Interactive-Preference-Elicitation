from matplotlib import pyplot as plt
import numpy as np
from pylab import mpl
import json
import time
import scipy.stats as stats
import math
import pandas as pd

# ---- added by fix_paths.py: make this script portable ----
import os as _os
CODE_ROOT = _os.environ.get(
    "CODE_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
# ---- end CODE_ROOT snippet ----


timestart = time.time()

# ==== load data ====
df = pd.read_csv(f"{CODE_ROOT}/results/section5/results.csv")
df = df[~(df["method"]=="MCTS_Entropy")]  
df = df[~(df["method"]=="MCTS_Var")]  
df["method"] = df["method"].replace("MCTS_Var_Cp_0.7071067811865475", "MCTS_Var")
df["method"] = df["method"].replace("MCTS_Entropy_Cp_0.7071067811865475", "MCTS_Ent")
df["method"] = df["method"].replace("DVF", "H_DVF")
df["method"] = df["method"].replace("RAND", "H_RAND")
df["method"] = df["method"].replace("E_PWI_d1", "H_PWI_d1")
df["method"] = df["method"].replace("E_PWI_d2", "H_PWI_d2")
df["method"] = df["method"].replace("E_RAI_d1", "H_RAI_d1")
df["method"] = df["method"].replace("E_RAI_d2", "H_RAI_d2")
df = df.rename(columns={"var": "fVAR"})


# Use random_index 0..19 for every (n, m, T) cell; no skip/replay.
df = df[df["random_index"] < 20]
print(df)

# sample count per method
print(df['method'].value_counts())

# one-sided Wilcoxon signed-rank test (paired by random_index, n, m, rounds)
def f(data1, data2):
    return np.sum(data1 < data2)

metrics = ["fVAR", "fPWI", "fRAI"]
for metric in metrics:
      data1 = np.array(df[df['method']=="MCTS_Var"][metric])
      data2 = np.array(df[df['method']=="MCTS_Ent"][metric])
      data3 = np.array(df[df['method']=="H_PWI_d1"][metric])
      data4 = np.array(df[df['method']=="H_PWI_d2"][metric])
      data5 = np.array(df[df['method']=="H_RAI_d1"][metric])
      data6 = np.array(df[df['method']=="H_RAI_d2"][metric])
      data7 = np.array(df[df['method']=="H_DVF"][metric])
      data8 = np.array(df[df['method']=="H_RAND"][metric])

      # fVAR
      print('& MCTS_Var &', '-', 
            '&', round(stats.wilcoxon(data1, data2, alternative="less")[1], 3), f"({f(data1, data2)})",
            '&', round(stats.wilcoxon(data1, data3, alternative="less")[1], 3), f"({f(data1, data3)})",
            '&', round(stats.wilcoxon(data1, data4, alternative="less")[1], 3), f"({f(data1, data4)})",
            '&', round(stats.wilcoxon(data1, data5, alternative="less")[1], 3), f"({f(data1, data5)})",
            '&', round(stats.wilcoxon(data1, data6, alternative="less")[1], 3), f"({f(data1, data6)})",
            '&', round(stats.wilcoxon(data1, data7, alternative="less")[1], 3), f"({f(data1, data7)})",
            '&', round(stats.wilcoxon(data1, data8, alternative="less")[1], 3), f"({f(data1, data8)})",'\\\\')

      print('& MCTS_Ent &', round(stats.wilcoxon(data2, data1, alternative="less")[1], 3), f"({f(data2, data1)})",
            '&', '-', 
            '&', round(stats.wilcoxon(data2, data3, alternative="less")[1], 3), f"({f(data2, data3)})",
            '&', round(stats.wilcoxon(data2, data4, alternative="less")[1], 3), f"({f(data2, data4)})",
            '&', round(stats.wilcoxon(data2, data5, alternative="less")[1], 3), f"({f(data2, data5)})",
            '&', round(stats.wilcoxon(data2, data6, alternative="less")[1], 3), f"({f(data2, data6)})",
            '&', round(stats.wilcoxon(data2, data7, alternative="less")[1], 3), f"({f(data2, data7)})",
            '&', round(stats.wilcoxon(data2, data8, alternative="less")[1], 3), f"({f(data2, data8)})",'\\\\')

      print('& $H_\\textit{PWI}$ &', round(stats.wilcoxon(data3, data1, alternative="less")[1], 3), f"({f(data3, data1)})",
            '&', round(stats.wilcoxon(data3, data2, alternative="less")[1], 3), f"({f(data3, data2)})",
            '&', '-', 
            '&', round(stats.wilcoxon(data3, data4, alternative="less")[1], 3), f"({f(data3, data4)})",
            '&', round(stats.wilcoxon(data3, data5, alternative="less")[1], 3), f"({f(data3, data5)})",
            '&', round(stats.wilcoxon(data3, data6, alternative="less")[1], 3), f"({f(data3, data6)})",
            '&', round(stats.wilcoxon(data3, data7, alternative="less")[1], 3), f"({f(data3, data7)})",
            '&', round(stats.wilcoxon(data3, data8, alternative="less")[1], 3), f"({f(data3, data8)})",'\\\\')
      
      print('& $H_\\textit{RAI}$ &', round(stats.wilcoxon(data4, data1, alternative="less")[1], 3), f"({f(data4, data1)})",
            '&', round(stats.wilcoxon(data4, data2, alternative="less")[1], 3), f"({f(data4, data2)})",
            '&', round(stats.wilcoxon(data4, data3, alternative="less")[1], 3), f"({f(data4, data3)})",
            '&', '-', 
            '&', round(stats.wilcoxon(data4, data5, alternative="less")[1], 3), f"({f(data4, data5)})",
            '&', round(stats.wilcoxon(data4, data6, alternative="less")[1], 3), f"({f(data4, data6)})",
            '&', round(stats.wilcoxon(data4, data7, alternative="less")[1], 3), f"({f(data4, data7)})",
            '&', round(stats.wilcoxon(data4, data8, alternative="less")[1], 3), f"({f(data4, data8)})",'\\\\')
      
      print('& $H_{\\textit{PWI}-2}$ &', round(stats.wilcoxon(data5, data1, alternative="less")[1], 3), f"({f(data5, data1)})",
            '&', round(stats.wilcoxon(data5, data2, alternative="less")[1], 3), f"({f(data5, data2)})",
            '&', round(stats.wilcoxon(data5, data3, alternative="less")[1], 3), f"({f(data5, data3)})",
            '&', round(stats.wilcoxon(data5, data4, alternative="less")[1], 3), f"({f(data5, data4)})",
            '&', '-', 
            '&', round(stats.wilcoxon(data5, data6, alternative="less")[1], 3), f"({f(data5, data6)})",
            '&', round(stats.wilcoxon(data5, data7, alternative="less")[1], 3), f"({f(data5, data7)})",
            '&', round(stats.wilcoxon(data5, data8, alternative="less")[1], 3), f"({f(data5, data8)})",'\\\\')
      
      print('& $H_{\\textit{RAI}-2}$ &', round(stats.wilcoxon(data6, data1, alternative="less")[1], 3), f"({f(data6, data1)})",
            '&', round(stats.wilcoxon(data6, data2, alternative="less")[1], 3), f"({f(data6, data2)})",
            '&', round(stats.wilcoxon(data6, data3, alternative="less")[1], 3), f"({f(data6, data3)})",
            '&', round(stats.wilcoxon(data6, data4, alternative="less")[1], 3), f"({f(data6, data4)})",
            '&', round(stats.wilcoxon(data6, data5, alternative="less")[1], 3), f"({f(data6, data5)})",
            '&', '-', 
            '&', round(stats.wilcoxon(data6, data7, alternative="less")[1], 3), f"({f(data6, data7)})",
            '&', round(stats.wilcoxon(data6, data8, alternative="less")[1], 3), f"({f(data6, data8)})",'\\\\')
      
      print('& $H_\\textit{DVF}$ &', round(stats.wilcoxon(data7, data1, alternative="less")[1], 3), f"({f(data7, data1)})",
            '&', round(stats.wilcoxon(data7, data2, alternative="less")[1], 3), f"({f(data7, data2)})",
            '&', round(stats.wilcoxon(data7, data3, alternative="less")[1], 3), f"({f(data7, data3)})",
            '&', round(stats.wilcoxon(data7, data4, alternative="less")[1], 3), f"({f(data7, data4)})",
            '&', round(stats.wilcoxon(data7, data5, alternative="less")[1], 3), f"({f(data7, data5)})",
            '&', round(stats.wilcoxon(data7, data6, alternative="less")[1], 3), f"({f(data7, data6)})",
            '&', '-', 
            '&', round(stats.wilcoxon(data7, data8, alternative="less")[1], 3), f"({f(data7, data8)})",'\\\\')
      
      print('& $H_\\textit{RAND}$ &', round(stats.wilcoxon(data8, data1, alternative="less")[1], 3), f"({f(data8, data1)})",
            '&', round(stats.wilcoxon(data8, data2, alternative="less")[1], 3), f"({f(data8, data2)})",
            '&', round(stats.wilcoxon(data8, data3, alternative="less")[1], 3), f"({f(data8, data3)})",
            '&', round(stats.wilcoxon(data8, data4, alternative="less")[1], 3), f"({f(data8, data4)})",
            '&', round(stats.wilcoxon(data8, data5, alternative="less")[1], 3), f"({f(data8, data5)})",
            '&', round(stats.wilcoxon(data8, data6, alternative="less")[1], 3), f"({f(data8, data6)})",
            '&', round(stats.wilcoxon(data8, data7, alternative="less")[1], 3), f"({f(data8, data7)})",
            '&', '-', '\\\\')

endtime = time.time()   
print("Total time (s):", endtime - timestart)