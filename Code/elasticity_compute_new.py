import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from utilities import get_lorenz_shares
from estimation import getDistributionsFromHetParamValues, calc_Lorenz_Sim
from parameters import opt_center, opt_spread

# === Capture all print output ===
import sys, io
_capture = io.StringIO()
_sys_stdout = sys.stdout
sys.stdout = _capture
# ================================

eff_R = 1 + 0.095

# =====================================================================
# UNIFORM DISTRIBUTION — INFINITE HORIZON (PY)
# =====================================================================
unif_r_list_PY_2004 = [
    np.float64(0.9618236382326008), np.float64(0.9813452633823873),
    np.float64(1.0008668885321736), np.float64(1.0203885136819602),
    np.float64(1.0399101388317467), np.float64(1.0594317639815332),
    np.float64(1.0789533891313197)
]
unif_epsilon_list_PY_2004 = [r / (eff_R - r) for r in unif_r_list_PY_2004]

print("=== Uniform Distribution (PY, Infinite Horizon, 2004) ===")
print("r values:", unif_r_list_PY_2004)
print("epsilon values:", unif_epsilon_list_PY_2004)
print()

# =====================================================================
# UNIFORM DISTRIBUTION — LIFE-CYCLE (LC)
# =====================================================================
unif_r_list_LC_2004 = [
    np.float64(0.9199759016053255), np.float64(0.9471944847833991),
    np.float64(0.9744130679614726), np.float64(1.0016316511395462),
    np.float64(1.0288502343176198), np.float64(1.0560688174956934),
    np.float64(1.083287400673767)
]
unif_epsilon_list_LC_2004 = [r / (eff_R - r) for r in unif_r_list_LC_2004]

print("=== Uniform Distribution (LC, Life-Cycle, 2004) ===")
print("r values:", unif_r_list_LC_2004)
print("epsilon values:", unif_epsilon_list_LC_2004)
print()

# =====================================================================
# LOGNORMAL DISTRIBUTION — INFINITE HORIZON (PY)
# =====================================================================
lognorm_r_list_PY_2004 = [
    np.float64(0.9763833515293711), np.float64(1.0007225934313486),
    np.float64(1.0144942385004752), np.float64(1.026405367297889),
    np.float64(1.038457995014501), np.float64(1.0527577014290397),
    np.float64(1.079185833711637)
]
lognorm_epsilon_list_PY_2004 = [r / (eff_R - r) for r in lognorm_r_list_PY_2004]

print("=== Lognormal Distribution (PY, Infinite Horizon, 2004) ===")
print("r values:", lognorm_r_list_PY_2004)
print("epsilon values:", lognorm_epsilon_list_PY_2004)
print()

# =====================================================================
# LOGNORMAL DISTRIBUTION — LIFE-CYCLE (LC)
# =====================================================================
lognorm_r_list_LC_2004 = [
    np.float64(0.9362603964316518), np.float64(0.971100644330824),
    np.float64(0.9910275661512888), np.float64(1.008370855711804),
    np.float64(1.0260212617954967), np.float64(1.047094395418578),
    np.float64(1.0864705321096157)
]
lognorm_epsilon_list_LC_2004 = [r / (eff_R - r) for r in lognorm_r_list_LC_2004]

print("=== Lognormal Distribution (LC, Life-Cycle, 2004) ===")
print("r values:", lognorm_r_list_LC_2004)
print("epsilon values:", lognorm_epsilon_list_LC_2004)
print()

# =====================================================================
# SAVE ALL PRINTED OUTPUT TO ../Results/elasticities.txt
# =====================================================================
sys.stdout = _sys_stdout

script_dir = Path(__file__).resolve().parent       # e.g., .../Code
results_dir = script_dir.parent / "Results"        # move up one directory
results_dir.mkdir(exist_ok=True)

out_path = results_dir / "elasticities_2004.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(_capture.getvalue())

print(f"Wrote elasticities to: {out_path}")
# =====================================================================

