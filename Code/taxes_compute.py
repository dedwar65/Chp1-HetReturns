from copy import deepcopy
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from utilities import (get_lorenz_shares)
from utilities_taxes import (getDistributionsFromHetParamValues)
from utilities_taxes import (extract_income_distribution,
                            make_Rfree_with_wealth_tax,
                            make_Rfree_with_capital_income_tax,
                            save_lorenz_side_by_side_from_results,
                            require_vfunc,
                            compute_newborn_EV_per_agent,
                            compute_population_newborn_welfare,
                            consumption_equivalent_delta)

# Import SCF Lorenz data from parameters.py
from parameters import (emp_lorenz, MyPopulation, LifeCycle,
                        HetTypeCount, DstnType, DstnParamMapping, BaseTypeCount, tag, TargetPercentiles)

# Specify the center and spread from a previous successful estimation.py run
# You can find these values in the Results directory

#PY
#center = 1.0211998134197764
#spread = 0.06728339598564156

#LC
#center = 1.02299795492603
#spread=0.05538760051509333

# Optional fallback: uncomment to import optimal center/spread from parameters
# from parameters import opt_center as _opt_center, opt_spread as _opt_spread
# center = _opt_center
# spread = _opt_spread
# if (center is None) or (spread is None):
#     raise RuntimeError("center/spread are not set. Run estimation.py to populate parameters.opt_center/opt_spread or set them manually here.")

print(f"Using center: {center}, spread: {spread}")
print(f"SCF Lorenz data: {emp_lorenz}")

# Step 1: Extract income distribution and calculate GDP target
# This will update the population with the given center/spread and extract distributions
GDP, WealthDstn, ProdDstn, WeightDstn = extract_income_distribution(center, spread)

# Calculate GDP properly using dot product with weights
GDP_target = 0.01 * GDP  # 1% of GDP

print(f"Aggregate Income (GDP): {GDP:.2f}")
print(f"Target Tax Revenue (1% of GDP): {GDP_target:.2f}")

# Get interest rates from the distribution (following the pattern from calculate_capital_income_tax_revenue)
dstn = DstnType(*DstnParamMapping(center, spread)).discretize(HetTypeCount)
interest_rates = dstn.atoms[0]

# Compute aggregates once for reporting
aggregate_wealth_once = float(np.dot(WealthDstn, WeightDstn))

# Step 2: Find tax rates that generate the target revenue
print("\nFinding wealth tax rate...")
# Build taxable base by excluding non-positive wealth and using same indices for weights
_taxable_indices = np.where(WealthDstn > 0)[0]
_taxable_wealth_dstn = WealthDstn[_taxable_indices]
_taxable_weight_dstn = WeightDstn[_taxable_indices]

# Taxable wealth base
taxable_aggregate_wealth = float(np.dot(_taxable_wealth_dstn, _taxable_weight_dstn))
print(f"Aggregate Wealth (all): {aggregate_wealth_once:,.2f}")
print(f"Taxable Aggregate Wealth (>0): {taxable_aggregate_wealth:,.2f}")

# Wealth tax rate from taxable base
wealth_tax_rate = GDP_target / taxable_aggregate_wealth if taxable_aggregate_wealth > 0 else 0.0
print(f"Wealth tax rate: {wealth_tax_rate:.6f}")

print("\nFinding capital income tax rate...")
# Compute per-observation interest rate array aligned with WealthDstn/WeightDstn
# Uses the same concatenation order used to build those arrays in estimation.py
r_blocks = []
for ag in MyPopulation:
    if LifeCycle:
        # history arrays shape: (T, N)
        a_hist = getattr(ag.history, 'get', None)
        # Fall back to attributes directly for safety
        aLvl_hist = ag.history['aLvl']
        T, N = aLvl_hist.shape
        if isinstance(ag.Rfree, list):
            r_by_age = [float(ag.Rfree[t]) for t in range(T)]
        else:
            r_by_age = [float(ag.Rfree)] * T
        r_block = np.repeat(r_by_age, N)  # length T*N, matches flatten order
    else:
        # state_now arrays length: N
        N = ag.state_now['aLvl'].size
        if isinstance(ag.Rfree, list):
            r_val = float(ag.Rfree[0])
        else:
            r_val = float(ag.Rfree)
        r_block = np.full(N, r_val)
    r_blocks.append(r_block)
r_arr = np.concatenate(r_blocks)

_capital_taxable_mask = (WealthDstn > 0) & (r_arr > 1.0)
_capital_taxable_wealth = WealthDstn[_capital_taxable_mask]
_capital_taxable_weights = WeightDstn[_capital_taxable_mask]
_capital_taxable_returns = (r_arr[_capital_taxable_mask] - 1.0)

aggregate_capital_income = float(np.dot(_capital_taxable_wealth * _capital_taxable_returns, _capital_taxable_weights))
print(f"Taxable Aggregate Capital Income (wealth>0 and r>1): {aggregate_capital_income:,.2f}")

# Capital income tax rate from taxable base
capital_income_tax_rate = GDP_target / aggregate_capital_income if aggregate_capital_income > 0 else 0.0
print(f"Capital income tax rate: {capital_income_tax_rate:.6f}")

# Step 3: Create two new populations with taxes applied
print("\nCreating taxed populations...")

# Create wealth tax population
MyPopulation_WT = []
for agent in MyPopulation:
    agent_copy = deepcopy(agent)
    RfreeFull = agent_copy.Rfree
    agent_copy.Rfree = make_Rfree_with_wealth_tax(RfreeFull, wealth_tax_rate, agent_copy.T_cycle)
    MyPopulation_WT.append(agent_copy)

# Create capital income tax population
MyPopulation_CIT = []
for agent in MyPopulation:
    agent_copy = deepcopy(agent)
    RfreeFull = agent_copy.Rfree
    agent_copy.Rfree = make_Rfree_with_capital_income_tax(RfreeFull, capital_income_tax_rate, agent_copy.T_cycle)
    MyPopulation_CIT.append(agent_copy)

print(f"Created {len(MyPopulation_WT)} agents with wealth tax")
print(f"Created {len(MyPopulation_CIT)} agents with capital income tax")

# Print average Rfree for baseline vs taxed populations before solve/sim
import numpy as _np
avg_Rfree_base = _np.mean([_np.mean(a.Rfree) for a in MyPopulation])
avg_Rfree_wt   = _np.mean([_np.mean(a.Rfree) for a in MyPopulation_WT])
avg_Rfree_cit  = _np.mean([_np.mean(a.Rfree) for a in MyPopulation_CIT])
print(f"Average Rfree (baseline): {avg_Rfree_base:.6f}")
print(f"Average Rfree (wealth tax): {avg_Rfree_wt:.6f}")
print(f"Average Rfree (capital income tax): {avg_Rfree_cit:.6f}")

# Solve and simulate taxed populations to ensure histories reflect modified returns
for pop_name, pop in [("WT", MyPopulation_WT), ("CIT", MyPopulation_CIT)]:
    for ag in pop:
        if not hasattr(ag, 'track_vars') or ('aLvl' not in ag.track_vars):
            ag.track_vars = ['aLvl','pLvl','WeightFac']
        ag.solve()
        try:
            ag.initialize_sim()
            ag.simulate()
        except Exception:
            try:
                ag.initialize_sim()
                ag.simulate()
            except Exception:
                pass

# Step 4: Extract wealth distributions for each case
def extract_wealth_distribution(population):
    """
    Extract wealth, income, and weight arrays from a population.
    Uses the same pattern as estimation.py lines 150-159.
    """
    from HARK.parallel import multi_thread_commands
    multi_thread_commands(population, ["solve()", "initialize_sim()", "simulate()"], num_jobs=1)

    if LifeCycle:
        IndWealthArray = np.concatenate([this_type.history['aLvl'].flatten() for this_type in population])
        IndProdArray = np.concatenate([this_type.history['pLvl'].flatten() for this_type in population])
        IndWeightArray = np.concatenate([this_type.history['WeightFac'].flatten() for this_type in population])
    else:
        IndWealthArray = np.concatenate([this_type.state_now['aLvl'] for this_type in population])
        IndProdArray = np.concatenate([this_type.state_now['pLvl'] for this_type in population])
        IndWeightArray = np.concatenate([this_type.state_now['WeightFac'] for this_type in population])

    return IndWealthArray, IndProdArray, IndWeightArray

# Extract distributions for each population
print("\nExtracting wealth distributions...")

WealthArray_Original, ProdArray_Original, WeightArray_Original = extract_wealth_distribution(MyPopulation)
WealthArray_WT, ProdArray_WT, WeightArray_WT = extract_wealth_distribution(MyPopulation_WT)
WealthArray_CIT, ProdArray_CIT, WeightArray_CIT = extract_wealth_distribution(MyPopulation_CIT)

print(f"Original population wealth array length: {len(WealthArray_Original)}")
print(f"Wealth tax population wealth array length: {len(WealthArray_WT)}")
print(f"Capital income tax population wealth array length: {len(WealthArray_CIT)}")

# Summary statistics
print(f"\nOriginal population average wealth: {np.average(WealthArray_Original, weights=WeightArray_Original):.2f}")
print(f"Wealth tax population average wealth: {np.average(WealthArray_WT, weights=WeightArray_WT):.2f}")
print(f"Capital income tax population average wealth: {np.average(WealthArray_CIT, weights=WeightArray_CIT):.2f}")

# Compute Lorenz shares for each simulated wealth distribution
pctiles = np.array(TargetPercentiles, dtype=float)
Lorenz_orig = get_lorenz_shares(WealthArray_Original, WeightArray_Original, percentiles=pctiles)
Lorenz_wt = get_lorenz_shares(WealthArray_WT, WeightArray_WT, percentiles=pctiles)
Lorenz_cit = get_lorenz_shares(WealthArray_CIT, WeightArray_CIT, percentiles=pctiles)

# Convert to plain floats for clean printing
Lorenz_orig_list = [float(x) for x in np.round(Lorenz_orig, 6)]
Lorenz_wt_list   = [float(x) for x in np.round(Lorenz_wt,   6)]
Lorenz_cit_list  = [float(x) for x in np.round(Lorenz_cit,  6)]
print("Lorenz (Original):", Lorenz_orig_list)
print("Lorenz (Wealth tax):", Lorenz_wt_list)
print("Lorenz (Capital income tax):", Lorenz_cit_list)
print("Lorenz (SCF Data):", [float(x) for x in np.round(emp_lorenz, 6)])

# Store the arrays for later use with proper structure for plotting
tax_analysis_results = {
    'wealth_tax_rate': wealth_tax_rate,
    'capital_income_tax_rate': capital_income_tax_rate,
    'GDP_target': GDP_target,
    'SCF_lorenz': emp_lorenz,
    'original': {
        'wealth': WealthArray_Original,
        'income': ProdArray_Original,
        'weights': WeightArray_Original,
        'lorenz': Lorenz_orig
    },
    'wealth_tax': {
        'wealth': WealthArray_WT,
        'income': ProdArray_WT,
        'weights': WeightArray_WT,
        'lorenz': Lorenz_wt
    },
    'capital_income_tax': {
        'wealth': WealthArray_CIT,
        'income': ProdArray_CIT,
        'weights': WeightArray_CIT,
        'lorenz': Lorenz_cit
    }
}

# Export results to Results_taxes/{tag_wo_tax}.txt
_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Results/Results_taxes')
os.makedirs(_results_dir, exist_ok=True)
_filename_tag = tag.replace('tax_', '').replace('tax', '')
_results_path = os.path.join(_results_dir, f"{_filename_tag}.txt")
_results_lines = []
_results_lines.append(f"Tag: {tag}\n")
_results_lines.append(f"Aggregate Income (GDP): {GDP:.6f}\n")
_results_lines.append(f"Target Tax Revenue (1% of GDP): {GDP_target:.6f}\n")
_results_lines.append(f"Aggregate Wealth (all): {aggregate_wealth_once:.6f}\n")
_results_lines.append(f"Taxable Aggregate Wealth (>0): {taxable_aggregate_wealth:.6f}\n")
_results_lines.append(f"Taxable Aggregate Capital Income: {aggregate_capital_income:.6f}\n")
_results_lines.append(f"Wealth tax rate: {wealth_tax_rate:.6f}\n")
_results_lines.append(f"Capital income tax rate: {capital_income_tax_rate:.6f}\n")
_results_lines.append(f"Percentiles: {list(pctiles)}\n")
_results_lines.append(f"Lorenz (Original): {Lorenz_orig_list}\n")
_results_lines.append(f"Lorenz (Wealth tax): {Lorenz_wt_list}\n")
_results_lines.append(f"Lorenz (Capital income tax): {Lorenz_cit_list}\n")
_results_lines.append(f"Lorenz (SCF Data): {[float(x) for x in np.round(emp_lorenz, 6)]}\n")
with open(_results_path, 'w', encoding='utf-8') as _f:
    _f.writelines(_results_lines)
print(f"Saved results to: {_results_path}")

print("\nTax analysis complete! Arrays stored in 'tax_analysis_results' dictionary.")
# Save Lorenz figures
try:
    out_path = save_lorenz_side_by_side_from_results(tax_analysis_results, tag, pctiles)
    print(f"Saved Lorenz figure to: {out_path}")
except Exception as e:
    print(f"Lorenz figure save failed: {e}")
print("You can now use these arrays to compute wealth distributions and Lorenz shares.")

# ===================== Newborn welfare computation and outputs =====================

# Build return-type weights from discretization
het_weights = dstn.pmv

# Validate value function availability for baseline agents (fail fast if missing)
for ag in MyPopulation:
    try:
        require_vfunc(ag)
    except RuntimeError as e:
        raise

# Compute per-type EVs and aggregates under each regime
W_orig_vec, W_orig_pmv, W_orig_counts = compute_population_newborn_welfare(MyPopulation, het_weights, BaseTypeCount)
W_wt_vec,   W_wt_pmv,   W_wt_counts   = compute_population_newborn_welfare(MyPopulation_WT, het_weights, BaseTypeCount)
W_cit_vec,  W_cit_pmv,  W_cit_counts  = compute_population_newborn_welfare(MyPopulation_CIT, het_weights, BaseTypeCount)

# Retrieve CRRA
rho = float(MyPopulation[0].CRRA)

# Aggregate CE comparisons (both weighting modes)
CE_WT_vs_CIT_pmv    = consumption_equivalent_delta(W_wt_pmv,  W_cit_pmv,  rho)
CE_WT_vs_ORG_pmv    = consumption_equivalent_delta(W_wt_pmv,  W_orig_pmv, rho)
CE_CIT_vs_ORG_pmv   = consumption_equivalent_delta(W_cit_pmv, W_orig_pmv, rho)

CE_WT_vs_CIT_counts = consumption_equivalent_delta(W_wt_counts,  W_cit_counts,  rho)
CE_WT_vs_ORG_counts = consumption_equivalent_delta(W_wt_counts,  W_orig_counts, rho)
CE_CIT_vs_ORG_counts= consumption_equivalent_delta(W_cit_counts, W_orig_counts, rho)

# Per-type CE comparisons (WT vs CIT)
CE_per_type_WT_vs_CIT = [consumption_equivalent_delta(W_wt_vec[i], W_cit_vec[i], rho) for i in range(len(W_orig_vec))]

# Save welfare results to Results_taxes/{tag}_welfare.txt
_welfare_path = os.path.join(_results_dir, f"{tag}_welfare.txt")
_wl = []
_wl.append(f"Tag: {tag}\n")
_wl.append(f"CRRA (rho): {rho:.6f}\n")
_wl.append(f"Wealth tax rate: {wealth_tax_rate:.6f}\n")
_wl.append(f"Capital income tax rate: {capital_income_tax_rate:.6f}\n\n")

_wl.append("Per-type newborn EVs (Original):\n")
_wl.append(f"{[float(x) for x in np.round(W_orig_vec, 8)]}\n")
_wl.append("Per-type newborn EVs (Wealth tax):\n")
_wl.append(f"{[float(x) for x in np.round(W_wt_vec, 8)]}\n")
_wl.append("Per-type newborn EVs (Capital income tax):\n")
_wl.append(f"{[float(x) for x in np.round(W_cit_vec, 8)]}\n\n")

_wl.append("Aggregate newborn welfare (pmv weights):\n")
_wl.append(f"Original: {W_orig_pmv:.8f}\n")
_wl.append(f"Wealth tax: {W_wt_pmv:.8f}\n")
_wl.append(f"Capital income tax: {W_cit_pmv:.8f}\n\n")

_wl.append("Aggregate newborn welfare (AgentCount shares):\n")
_wl.append(f"Original: {W_orig_counts:.8f}\n")
_wl.append(f"Wealth tax: {W_wt_counts:.8f}\n")
_wl.append(f"Capital income tax: {W_cit_counts:.8f}\n\n")

_wl.append("Consumption-equivalent (pmv weights), Δ (e.g., 0.07 = +7%):\n")
_wl.append(f"WT vs CIT: {CE_WT_vs_CIT_pmv:.8f}\n")
_wl.append(f"WT vs Original: {CE_WT_vs_ORG_pmv:.8f}\n")
_wl.append(f"CIT vs Original: {CE_CIT_vs_ORG_pmv:.8f}\n\n")

_wl.append("Consumption-equivalent (AgentCount shares), Δ:\n")
_wl.append(f"WT vs CIT: {CE_WT_vs_CIT_counts:.8f}\n")
_wl.append(f"WT vs Original: {CE_WT_vs_ORG_counts:.8f}\n")
_wl.append(f"CIT vs Original: {CE_CIT_vs_ORG_counts:.8f}\n\n")

_wl.append("Per-type consumption-equivalent WT vs CIT, Δ by return type (low→high):\n")
_wl.append(f"{[float(x) for x in np.round(CE_per_type_WT_vs_CIT, 8)]}\n")

with open(_welfare_path, 'w', encoding='utf-8') as _wf:
    _wf.writelines(_wl)
print(f"Saved newborn welfare results to: {_welfare_path}")
