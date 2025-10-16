# -*- coding: utf-8 -*-
# Generates "MPCs by Wealth Deciles" comparing hetero vs no-hetero returns.
# Saves both PNG and PDF to ../Tables/ regardless of where you run it from.

from pathlib import Path
import matplotlib.pyplot as plt

# 1. ─── MPC by Wealth Decile (PY) ─────────────────────────────────────────────────────
# -----------------
# Data
# -----------------
deciles = [
    "0–10", "10–20", "20–30", "30–40", "40–50",
    "50–60", "60–70", "70–80", "80–90", "90–100"
]

# No heterogeneity case
mpc_nohet = [
    0.364255, 0.148613, 0.104436, 0.086638, 0.077941,
    0.073416, 0.071035, 0.069164, 0.067761, 0.065852
]

# Heterogeneous returns case
mpc_het = [
    0.729528, 0.437812, 0.371712, 0.308848, 0.254595,
    0.216160, 0.170530, 0.117608, 0.074096, 0.065101
]

# -----------------
# Paths
# -----------------
script_path = Path(__file__).resolve()
repo_root = script_path.parent.parent
tables_dir = repo_root / "Tables"
tables_dir.mkdir(parents=True, exist_ok=True)

png_path = tables_dir / "Unif_PY_MPC_by_WealthDecile_compare.png"
pdf_path = tables_dir / "Unif_PY_MPC_by_WealthDecile_compare.pdf"

# -----------------
# Plot
# -----------------
plt.figure(figsize=(7, 4))
plt.plot(deciles, mpc_nohet, marker="o", color="tab:blue", linewidth=2, label="No het.")
plt.plot(deciles, mpc_het, marker="s", color="tab:orange", linewidth=2, label="Het. returns")

plt.title("MPCs by Wealth Deciles (PY)", fontsize=14)
plt.xlabel("Wealth-to-Income Percentile", fontsize=12)
plt.ylabel("Average MPC", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# Legend: upper right, semi-transparent background
plt.legend(frameon=True, loc="upper right", fontsize=10, facecolor="white", framealpha=0.9)

plt.tight_layout()

# -----------------
# Save
# -----------------
plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
# ─── MPC by Wealth Decile (PY) ─────────────────────────────────────────────────────


# 2. ─── MPC by Wealth Decile (LC) ─────────────────────────────────────────────────────

# -----------------
# Data
# -----------------
deciles = [
    "0–10", "10–20", "20–30", "30–40", "40–50",
    "50–60", "60–70", "70–80", "80–90", "90–100"
]

# No heterogeneity (LC)
mpc_nohet_lc = [
    0.391550, 0.180420, 0.110934, 0.085039, 0.078103,
    0.078063, 0.079955, 0.082561, 0.085600, 0.087293
]

# Heterogeneous returns (LC)
mpc_het_lc = [
    0.737631, 0.467820, 0.435019, 0.359397, 0.278068,
    0.204996, 0.118457, 0.077758, 0.073197, 0.087226
]

# -----------------
# Paths
# -----------------
script_path = Path(__file__).resolve()
repo_root = script_path.parent.parent
tables_dir = repo_root / "Tables"
tables_dir.mkdir(parents=True, exist_ok=True)

png_path = tables_dir / "Unif_LC_MPC_by_WealthDecile_compare.png"
pdf_path = tables_dir / "UNif_LC_MPC_by_WealthDecile_compare.pdf"

# -----------------
# Plot
# -----------------
plt.figure(figsize=(7, 4))
plt.plot(deciles, mpc_nohet_lc, marker="o", color="tab:blue", linewidth=2, label="No het. (LC)")
plt.plot(deciles, mpc_het_lc, marker="s", color="tab:orange", linewidth=2, label="Het. returns (LC)")

plt.title("MPCs by Wealth Deciles (LC)", fontsize=14)
plt.xlabel("Wealth-to-Income Percentile", fontsize=12)
plt.ylabel("Average MPC", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# Legend
plt.legend(frameon=True, loc="upper right", fontsize=10, facecolor="white", framealpha=0.9)

plt.tight_layout()

# -----------------
# Save
# -----------------
plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
# ─── MPC by Wealth Decile (LC) ─────────────────────────────────────────────────────

# 3. ─── LOGNORMAL MPC by Wealth Decile (PY & LC) ─────────────────────────────────────────────────────

# -----------------
# Data
# -----------------
deciles = [
    "0–10", "10–20", "20–30", "30–40", "40–50",
    "50–60", "60–70", "70–80", "80–90", "90–100"
]

# Lognormal (PY)
mpc_lognorm_py = [
    0.663118, 0.409280, 0.343136, 0.275242, 0.230803,
    0.199444, 0.167301, 0.128914, 0.085973, 0.065416
]

# Lognormal (LC)
mpc_lognorm_lc = [
    0.699159, 0.447697, 0.391770, 0.321662, 0.253435,
    0.184624, 0.121835, 0.085033, 0.073984, 0.085042
]

# -----------------
# Paths (robust to where you run this)
# -----------------
script_path = Path(__file__).resolve()
repo_root = script_path.parent.parent
tables_dir = repo_root / "Tables"
tables_dir.mkdir(parents=True, exist_ok=True)

png_path = tables_dir / "Lognorm_PY_LC_MPC_by_WealthDecile_compare.png"
pdf_path = tables_dir / "Lognorm_PY_LC_MPC_by_WealthDecile_compare.pdf"

# -----------------
# Plot
# -----------------
plt.figure(figsize=(7, 4))

# PY first (so it appears first in legend)
plt.plot(deciles, mpc_lognorm_py, marker="o", linewidth=2,
         color="tab:blue", label="Lognormal (PY)")

plt.plot(deciles, mpc_lognorm_lc, marker="s", linewidth=2,
         color="tab:orange", label="Lognormal (LC)")

plt.title("MPCs by Wealth Deciles — Lognormal Returns Dist.", fontsize=14)
plt.xlabel("Wealth-to-Income Percentile", fontsize=12)
plt.ylabel("Average MPC", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# Legend out of the way
plt.legend(frameon=True, loc="upper right", fontsize=10,
           facecolor="white", framealpha=0.9)

plt.tight_layout()

# -----------------
# Save
# -----------------
plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
# ─── LOGNORMAL MPC by Wealth Decile (PY & LC) ─────────────────────────────────────────────────────
