"""
Generate Lorenz curves for liquid, nonfinancial, and net worth assets
using SCF processed data, and print Lorenz shares at selected percentiles.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utilities import plot_lorenz_from_data, get_lorenz_shares

# ─── Directories ─────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
DATA_DIR = script_dir.parent / 'Data'
FIGURES_DIR = script_dir.parent / 'Figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ─── Configuration ───────────────────────────────────────────────────────────
YEAR = 2004

# ─── Load and filter SCF processed CSV ─────────────────────────────────────────
csv_path = DATA_DIR / 'scf_processed_dc_2.csv'
df = pd.read_csv(csv_path)
if 'year' in df.columns:
    df = df[df['year'] == YEAR]
else:
    raise KeyError("Column 'year' not found in processed CSV; cannot filter by year.")

# ─── Extract asset columns and weights ────────────────────────────────────────
liq_vals = df['liq'].to_numpy()
nfin_vals = df['nfin'].to_numpy()
netw_vals = df['networth'].to_numpy()
weights = df['wgt'].to_numpy()

# ─── Define percentiles of interest ──────────────────────────────────────────
percentiles = np.array([0.2, 0.4, 0.6, 0.8])  # 20%, 40%, 60%, 80%

# ─── Compute and print Lorenz shares at percentiles ──────────────────────────
assets = [liq_vals, nfin_vals, netw_vals]
labels = ['Liquid Assets', 'Nonfinancial Assets', 'Net Worth']

print(f"Lorenz shares for SCF {YEAR} at percentiles:")
for asset, label in zip(assets, labels):
    shares = get_lorenz_shares(asset, weights, percentiles)
    share_str = ", ".join(f"{int(p*100)}%: {s:.4f}" for p, s in zip(percentiles, shares))
    print(f"  {label}: {share_str}")

# ─── Plot combined Lorenz curves ─────────────────────────────────────────────
plot_lorenz_from_data(
    assets=assets,
    weights=weights,
    labels=labels,
    percentiles=np.linspace(0.001, 0.999, 15),
    title=f"SCF {YEAR} Asset Class Lorenz Curves",
    output_dir=str(FIGURES_DIR)
)

# ─── Rename plot file to include the year ─────────────────────────────────────
original_file = FIGURES_DIR / 'lorenz_only.png'
new_file = FIGURES_DIR / f'lorenz_only_{YEAR}.png'
if original_file.exists():
    original_file.rename(new_file)
    print(f"Saved Lorenz plot as {new_file}")
else:
    print(f"Expected plot file not found: {original_file}")

