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
                            save_lorenz_side_by_side_from_results)

# Import SCF Lorenz data from parameters.py
from parameters import (emp_lorenz, MyPopulation, LifeCycle,
                        HetTypeCount, DstnType, DstnParamMapping, BaseTypeCount, tag, TargetPercentiles)

# Specify the center and spread from a previous successful estimation.py run
# You can find these values in the Results directory

#UNIFORM DIST
#PY
center=1.0203885136819602
spread=0.0683256880242527


#LC
#center=1.0016316511395462
#spread=0.09526504112325755


#Lognormal DIST
#PY
#center=1.0269152972734659
#spread=0.03168873649157252


#LC
#center=1.0094779502784657
#spread=0.047100563599260487



print(f"Using center: {center}, spread: {spread}")
print(f"SCF Lorenz data: {emp_lorenz}")

# Step 1: Extract income distribution and calculate GDP target
# This will update the population with the given center/spread and extract distributions
GDP, WealthDstn, ProdDstn, WeightDstn = extract_income_distribution(center, spread)

# Target percentiles for Lorenz
pctiles = np.array(TargetPercentiles, dtype=float)
Lorenz_orig = get_lorenz_shares(WealthDstn, WeightDstn, percentiles=pctiles)

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

# === DIAGNOSTICS: Capital-income taxable base verification ===
_eps = 1e-12

print("\n[Diagnostics] Distribution array shapes and basic checks:")
print(f"len(WealthDstn)={len(WealthDstn)}, len(ProdDstn)={len(ProdDstn)}, len(WeightDstn)={len(WeightDstn)}, len(r_arr)={len(r_arr)}")

import numpy as _np
for name, arr in [("WealthDstn", WealthDstn), ("ProdDstn", ProdDstn), ("WeightDstn", WeightDstn), ("r_arr", r_arr)]:
    _has_nan = _np.isnan(arr).any()
    _has_inf = _np.isinf(arr).any()
    print(f"{name}: has_nan={_has_nan}, has_inf={_has_inf}")

w_min, w_max = float(_np.min(WealthDstn)), float(_np.max(WealthDstn))
r_min, r_max = float(_np.min(r_arr)), float(_np.max(r_arr))
print(f"Wealth min/max: {w_min:.12f} / {w_max:.12f}")
print(f"Return min/max: {r_min:.12f} / {r_max:.12f}")

count_w_le0     = int(_np.sum(WealthDstn <= 0.0))
count_w_gt0     = int(_np.sum(WealthDstn > 0.0))
count_r_le1_eps = int(_np.sum(r_arr <= 1.0 + _eps))
count_r_gt1_eps = int(_np.sum(r_arr > 1.0 + _eps))
print(f"Count wealth<=0: {count_w_le0}, wealth>0: {count_w_gt0}")
print(f"Count r<=1+eps: {count_r_le1_eps}, r>1+eps: {count_r_gt1_eps}")

_mask_wpos      = (WealthDstn > 0.0)
_mask_rpos_inc  = (r_arr > 1.0 + _eps)
_mask_taxable   = _mask_wpos & _mask_rpos_inc
print(f"Taxable mask count: {int(_np.sum(_mask_taxable))}")

agg_wealth_all  = float(_np.dot(WealthDstn, WeightDstn))
agg_wealth_pos  = float(_np.dot(WealthDstn[_mask_wpos], WeightDstn[_mask_wpos]))
print(f"Aggregate wealth (all): {agg_wealth_all:.12f}")
print(f"Aggregate wealth (wealth>0): {agg_wealth_pos:.12f}")
print(f"Difference: {(agg_wealth_all - agg_wealth_pos):.12f}")

_cap_base_masked = float(_np.dot((WealthDstn[_mask_taxable] * (r_arr[_mask_taxable] - 1.0)), WeightDstn[_mask_taxable]))
r_excess = _np.maximum(r_arr - 1.0, 0.0)
_cap_base_relu = float(_np.dot((WealthDstn[_mask_wpos] * r_excess[_mask_wpos]), WeightDstn[_mask_wpos]))

print("Taxable Aggregate Capital Income (masked): {:.12f}".format(float(_cap_base_masked)))
print("Taxable Aggregate Capital Income (ReLU):   {:.12f}".format(float(_cap_base_relu)))
print("Difference: {:.12f}".format(float(_cap_base_masked - _cap_base_relu)))

close_to_one = int(_np.sum(_np.isclose(r_arr, 1.0, atol=1e-8)))
print(f"Count r approximately 1 (|r-1|<=1e-8): {close_to_one}")

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

# Immediately verify per-type returns mapping (baseline vs taxed), using agent lists
def _collect_type_r(pop):
    vals = []
    for i in range(HetTypeCount):
        ag = pop[i * BaseTypeCount]
        r = ag.Rfree
        if isinstance(r, list):
            vals.append(float(r[0]))
        else:
            vals.append(float(r))
    return vals

baseline_R_types = _collect_type_r(MyPopulation)
wt_R_types       = _collect_type_r(MyPopulation_WT)
cit_R_types      = _collect_type_r(MyPopulation_CIT)

print("Per-type gross returns (low→high by type index):")
print(f"Baseline: {baseline_R_types}")
print(f"Wealth tax: {wt_R_types}")
print(f"Capital income tax: {cit_R_types}")

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

# Step 4: Extract wealth distributions for each case (simulate all regimes consistently)
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
print("\nExtracting wealth distributions for baseline and taxed populations...")

# Baseline: ensure baseline population is solved/simulated the same way
WealthArray_BASE, ProdArray_BASE, WeightArray_BASE = extract_wealth_distribution(MyPopulation)

WealthArray_WT, ProdArray_WT, WeightArray_WT = extract_wealth_distribution(MyPopulation_WT)
WealthArray_CIT, ProdArray_CIT, WeightArray_CIT = extract_wealth_distribution(MyPopulation_CIT)

print(f"Baseline population wealth array length: {len(WealthArray_BASE)}")
print(f"Wealth tax population wealth array length: {len(WealthArray_WT)}")
print(f"Capital income tax population wealth array length: {len(WealthArray_CIT)}")

# Summary statistics
print(f"\nOriginal population average wealth (simulated): {np.average(WealthArray_BASE, weights=WeightArray_BASE):.2f}")
print(f"Wealth tax population average wealth: {np.average(WealthArray_WT, weights=WeightArray_WT):.2f}")
print(f"Capital income tax population average wealth: {np.average(WealthArray_CIT, weights=WeightArray_CIT):.2f}")

# Compute Lorenz shares for all regimes from simulated arrays
Lorenz_base = get_lorenz_shares(WealthArray_BASE, WeightArray_BASE, percentiles=pctiles)
Lorenz_wt   = get_lorenz_shares(WealthArray_WT,   WeightArray_WT,   percentiles=pctiles)
Lorenz_cit  = get_lorenz_shares(WealthArray_CIT,  WeightArray_CIT,  percentiles=pctiles)


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
        'wealth': WealthDstn,
        'income': ProdDstn,
        'weights': WeightDstn,
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

# ===== Custom Value Function Welfare Analysis =====
from value_functions import (
    compute_value_function_bellman,
    create_value_function_interpolator,
    compute_newborn_EV_PY,
    compute_population_welfare_custom,
    ih_discount_sum_custom,
    consumption_equivalent_delta_custom,
    diagnose_agent_value_computation,
    compute_value_functions_lc,
    compute_newborn_EV_LC,
    lc_discount_sum_custom
)

# Set up m_grid for value function computation
m_grid = np.linspace(0.01, 20.0, 200)

print("\nComputing welfare using custom value functions...")
het_weights = dstn.pmv

# Compute welfare for each regime (PY focus)
W_orig_vec, W_orig_pmv, W_orig_counts = compute_population_welfare_custom(
    MyPopulation, het_weights, BaseTypeCount, m_grid
)
W_wt_vec, W_wt_pmv, W_wt_counts = compute_population_welfare_custom(
    MyPopulation_WT, het_weights, BaseTypeCount, m_grid
)
W_cit_vec, W_cit_pmv, W_cit_counts = compute_population_welfare_custom(
    MyPopulation_CIT, het_weights, BaseTypeCount, m_grid
)

rho = float(MyPopulation[0].CRRA)

_welfare_path = os.path.join(_results_dir, f"{tag}_welfare.txt")
_wl = []
_wl.append(f"Tag: {tag}\n")
_wl.append(f"CRRA (rho): {rho:.6f}\n")
_wl.append(f"Wealth tax rate: {wealth_tax_rate:.6f}\n")
_wl.append(f"Capital income tax rate: {capital_income_tax_rate:.6f}\n\n")

if abs(rho - 1.0) < 1e-12:
    # Per-type S values and aggregates
    S_per_type = [ih_discount_sum_custom(MyPopulation[i * BaseTypeCount])
                  for i in range(len(W_orig_vec))]
    import numpy as _np
    S_pmv = float(_np.dot(_np.array(S_per_type), _np.array(het_weights)))

    # CE comparisons (pmv)
    CE_WT_vs_CIT_pmv = consumption_equivalent_delta_custom(W_wt_pmv, W_cit_pmv, rho, S_pmv)
    CE_WT_vs_ORG_pmv = consumption_equivalent_delta_custom(W_wt_pmv, W_orig_pmv, rho, S_pmv)
    CE_CIT_vs_ORG_pmv = consumption_equivalent_delta_custom(W_cit_pmv, W_orig_pmv, rho, S_pmv)

    # Per-type CE (WT vs CIT)
    CE_per_type_WT_vs_CIT = [
        consumption_equivalent_delta_custom(W_wt_vec[i], W_cit_vec[i], rho, S_per_type[i])
        for i in range(len(W_orig_vec))
    ]
else:
    CE_WT_vs_CIT_pmv = consumption_equivalent_delta_custom(W_wt_pmv, W_cit_pmv, rho)
    CE_WT_vs_ORG_pmv = consumption_equivalent_delta_custom(W_wt_pmv, W_orig_pmv, rho)
    CE_CIT_vs_ORG_pmv = consumption_equivalent_delta_custom(W_cit_pmv, W_orig_pmv, rho)

    # Per-type CE (WT vs CIT)
    CE_per_type_WT_vs_CIT = [
        consumption_equivalent_delta_custom(W_wt_vec[i], W_cit_vec[i], rho)
        for i in range(len(W_orig_vec))
    ]

if not LifeCycle:
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

    _wl.append("Consumption-equivalent (pmv weights), Δ (e.g., 0.07 = +7%):\n")
    _wl.append(f"WT vs CIT: {CE_WT_vs_CIT_pmv:.8f}\n")
    _wl.append(f"WT vs Original: {CE_WT_vs_ORG_pmv:.8f}\n")
    _wl.append(f"CIT vs Original: {CE_CIT_vs_ORG_pmv:.8f}\n\n")

    _wl.append("Per-type consumption-equivalent WT vs CIT, Δ by return type (low→high):\n")
    _wl.append(f"{[float(x) for x in np.round(CE_per_type_WT_vs_CIT, 8)]}\n")

    _wl.append("\nPer-type returns by return type (low→high):\n")
    _wl.append(f"Baseline R: {[float(x) for x in np.round(baseline_R_types, 8)]}\n")
    _wl.append(f"Wealth tax R: {[float(x) for x in np.round(wt_R_types, 8)]}\n")
    _wl.append(f"Capital income tax R: {[float(x) for x in np.round(cit_R_types, 8)]}\n")

with open(_welfare_path, 'w', encoding='utf-8') as _wf:
    _wf.writelines(_wl)
print(f"Saved newborn welfare results to: {_welfare_path}")

# ===== Diagnostics: Value function and EV checks =====
try:
    # Representative agent (lowest return type, first base type)
    rep_indices = [0, BaseTypeCount, 2*BaseTypeCount]
    regimes = [
        ("Original", MyPopulation),
        ("Wealth tax", MyPopulation_WT),
        ("Capital income tax", MyPopulation_CIT)
    ]

    _diag_lines = []
    _diag_lines.append("\n[Diagnostics] Value function and EV checks (representative agents)\n")
    for name, pop in regimes:
        idx = rep_indices[0]
        ag = pop[idx]
        diag = diagnose_agent_value_computation(ag, m_grid)
        print(f"[Diagnostics] {name}: EV_newborn={diag['EV_newborn']:.6f}, V_min={diag['V_min']:.6f}, V_max={diag['V_max']:.6f}, V_mean={diag['V_mean']:.6f}")
        _diag_lines.append(f"{name}: EV_newborn={diag['EV_newborn']:.8f}, V_min={diag['V_min']:.8f}, V_max={diag['V_max']:.8f}, V_mean={diag['V_mean']:.8f}\n")
        _diag_lines.append(f"  Params: rho={diag['rho']:.6f}, beta={diag['beta']:.6f}, R={diag['R']:.6f}, Gamma={diag['Gamma']:.6f}, LivPrb={diag['LivPrb']:.6f}, psi_mean={diag['psi_mean']:.6f}, theta_mean={diag['theta_mean']:.6f}\n")
        _diag_lines.append(f"  c(m) samples: {diag['c_samples']}\n")

    if not LifeCycle:
        with open(_welfare_path, 'a', encoding='utf-8') as _wf:
            _wf.writelines(_diag_lines)
        print("Diagnostics appended to welfare file.")
except Exception as e:
    print(f"Diagnostics failed: {e}")

# ===== Life-Cycle Welfare (per education), if applicable =====
if LifeCycle:
    try:
        edu_labels = ["NoHS", "HS", "College"]

        def _evs_by_edu(pop):
            evs = {lab: [] for lab in edu_labels}
            for i in range(HetTypeCount):
                for j, lab in enumerate(edu_labels):
                    ag = pop[i * BaseTypeCount + j]
                    V_funcs, _ = compute_value_functions_lc(ag, m_grid)
                    evs[lab].append(compute_newborn_EV_LC(ag, V_funcs))
            return evs

        ev_orig_by_edu = _evs_by_edu(MyPopulation)
        ev_wt_by_edu   = _evs_by_edu(MyPopulation_WT)
        ev_cit_by_edu  = _evs_by_edu(MyPopulation_CIT)

        ret_weights = list(map(float, dstn.pmv))
        nohs_frac, hs_frac, college_frac = 0.11, 0.54, 0.35
        edu_fracs = {"NoHS": nohs_frac, "HS": hs_frac, "College": college_frac}

        import numpy as _np
        def _agg_ev(evs_by_edu, weights_returns, weights_edu):
            ev_edu = {lab: float(_np.dot(_np.array(evs_by_edu[lab], dtype=float), _np.array(weights_returns, dtype=float))) for lab in edu_labels}
            W_pmv = float(sum(ev_edu[lab] * weights_edu[lab] for lab in edu_labels))
            return ev_edu, W_pmv

        ev_edu_orig, W_orig_pmv_lc = _agg_ev(ev_orig_by_edu, ret_weights, edu_fracs)
        ev_edu_wt,   W_wt_pmv_lc   = _agg_ev(ev_wt_by_edu,   ret_weights, edu_fracs)
        ev_edu_cit,  W_cit_pmv_lc  = _agg_ev(ev_cit_by_edu,  ret_weights, edu_fracs)

        def _collect_agent_counts_by_edu(pop):
            counts = {lab: 0.0 for lab in edu_labels}
            for i in range(HetTypeCount):
                for j, lab in enumerate(edu_labels):
                    ag = pop[i * BaseTypeCount + j]
                    counts[lab] += float(getattr(ag, 'AgentCount', 0.0))
            total = sum(counts.values())
            shares = {lab: (counts[lab] / total if total > 0 else float('nan')) for lab in edu_labels}
            return counts, shares

        base_counts, base_shares = _collect_agent_counts_by_edu(MyPopulation)
        W_orig_counts_lc = float(sum(ev_edu_orig[lab] * base_shares[lab] for lab in edu_labels))
        W_wt_counts_lc   = float(sum(ev_edu_wt[lab]   * base_shares[lab] for lab in edu_labels))
        W_cit_counts_lc  = float(sum(ev_edu_cit[lab]  * base_shares[lab] for lab in edu_labels))

        rho = float(MyPopulation[0].CRRA)
        if abs(rho - 1.0) < 1e-12:
            S_by_edu = {}
            for j, lab in enumerate(edu_labels):
                rep_ag = MyPopulation[0 * BaseTypeCount + j]
                S_by_edu[lab] = lc_discount_sum_custom(rep_ag)
            S_pmv_lc    = float(sum(S_by_edu[lab] * edu_fracs[lab] for lab in edu_labels))
            CE_WT_vs_CIT_pmv_lc     = consumption_equivalent_delta_custom(W_wt_pmv_lc,  W_cit_pmv_lc,  rho, S_pmv_lc)
            CE_WT_vs_ORG_pmv_lc     = consumption_equivalent_delta_custom(W_wt_pmv_lc,  W_orig_pmv_lc, rho, S_pmv_lc)
            CE_CIT_vs_ORG_pmv_lc    = consumption_equivalent_delta_custom(W_cit_pmv_lc, W_orig_pmv_lc, rho, S_pmv_lc)
            # Per-education per-type CE (WT vs CIT)
            CE_by_edu = {lab: [] for lab in edu_labels}
            for j, lab in enumerate(edu_labels):
                for i in range(HetTypeCount):
                    CE_by_edu[lab].append(
                        consumption_equivalent_delta_custom(
                            ev_wt_by_edu[lab][i], ev_cit_by_edu[lab][i], rho, S_by_edu[lab]
                        )
                    )
        else:
            CE_WT_vs_CIT_pmv_lc     = consumption_equivalent_delta_custom(W_wt_pmv_lc,  W_cit_pmv_lc,  rho)
            CE_WT_vs_ORG_pmv_lc     = consumption_equivalent_delta_custom(W_wt_pmv_lc,  W_orig_pmv_lc, rho)
            CE_CIT_vs_ORG_pmv_lc    = consumption_equivalent_delta_custom(W_cit_pmv_lc, W_orig_pmv_lc, rho)
            # Per-education per-type CE (WT vs CIT)
            CE_by_edu = {lab: [] for lab in edu_labels}
            for j, lab in enumerate(edu_labels):
                for i in range(HetTypeCount):
                    CE_by_edu[lab].append(
                        consumption_equivalent_delta_custom(
                            ev_wt_by_edu[lab][i], ev_cit_by_edu[lab][i], rho
                        )
                    )

        _wl_lc = []
        _wl_lc.append("\n[Life-Cycle] Per-education newborn EVs (Original):\n")
        for lab in edu_labels:
            _wl_lc.append(f"{lab}: {[float(x) for x in np.round(ev_orig_by_edu[lab], 8)]}\n")
        _wl_lc.append("[Life-Cycle] Per-education newborn EVs (Wealth tax):\n")
        for lab in edu_labels:
            _wl_lc.append(f"{lab}: {[float(x) for x in np.round(ev_wt_by_edu[lab], 8)]}\n")
        _wl_lc.append("[Life-Cycle] Per-education newborn EVs (Capital income tax):\n")
        for lab in edu_labels:
            _wl_lc.append(f"{lab}: {[float(x) for x in np.round(ev_cit_by_edu[lab], 8)]}\n")

        _wl_lc.append("\n[Life-Cycle] Aggregate newborn welfare (pmv using declared edu fractions):\n")
        _wl_lc.append(f"Original: {W_orig_pmv_lc:.8f}\n")
        _wl_lc.append(f"Wealth tax: {W_wt_pmv_lc:.8f}\n")
        _wl_lc.append(f"Capital income tax: {W_cit_pmv_lc:.8f}\n")

        _wl_lc.append("[Life-Cycle] Aggregate newborn welfare (AgentCount shares):\n")
        _wl_lc.append(f"Original: {W_orig_counts_lc:.8f}\n")
        _wl_lc.append(f"Wealth tax: {W_wt_counts_lc:.8f}\n")
        _wl_lc.append(f"Capital income tax: {W_cit_counts_lc:.8f}\n")

        _wl_lc.append("\n[Life-Cycle] Aggregate CE (pmv), Δ:\n")
        _wl_lc.append(f"WT vs CIT: {CE_WT_vs_CIT_pmv_lc:.8f}\n")
        _wl_lc.append(f"WT vs Original: {CE_WT_vs_ORG_pmv_lc:.8f}\n")
        _wl_lc.append(f"CIT vs Original: {CE_CIT_vs_ORG_pmv_lc:.8f}\n")

        # Append per-education per-type CE list (WT vs CIT)
        _wl_lc.append("\n[Life-Cycle] Per-education per-type CE (WT vs CIT), Δ by return type (low→high):\n")
        for lab in edu_labels:
            _wl_lc.append(f"{lab}: {[float(x) for x in np.round(CE_by_edu[lab], 8)]}\n")

        # Append per-education baseline and taxed returns by type (t=0)
        def _collect_type_r_by_edu(pop):
            per_edu = {lab: [] for lab in edu_labels}
            for i in range(HetTypeCount):
                for j, lab in enumerate(edu_labels):
                    ag = pop[i * BaseTypeCount + j]
                    r = ag.Rfree
                    r_val = float(r[0]) if isinstance(r, list) else float(r)
                    per_edu[lab].append(r_val)
            return per_edu

        base_r_by_edu = _collect_type_r_by_edu(MyPopulation)
        wt_r_by_edu   = _collect_type_r_by_edu(MyPopulation_WT)
        cit_r_by_edu  = _collect_type_r_by_edu(MyPopulation_CIT)

        _wl_lc.append("\n[Life-Cycle] Per-education returns by type (t=0), low→high:\n")
        for lab in edu_labels:
            _wl_lc.append(f"{lab} Baseline R: {[float(x) for x in np.round(base_r_by_edu[lab], 8)]}\n")
            _wl_lc.append(f"{lab} Wealth tax R: {[float(x) for x in np.round(wt_r_by_edu[lab], 8)]}\n")
            _wl_lc.append(f"{lab} Capital income tax R: {[float(x) for x in np.round(cit_r_by_edu[lab], 8)]}\n")

        with open(_welfare_path, 'a', encoding='utf-8') as _wf:
            _wf.writelines(_wl_lc)
        print("Life-cycle welfare section appended to welfare file.")
    except Exception as e:
        print(f"Life-cycle welfare failed: {e}")
