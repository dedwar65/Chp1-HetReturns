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
                            compute_newborn_EV_per_agent_lc,
                            compute_population_newborn_welfare,
                            consumption_equivalent_delta,
                            consumption_equivalent_delta_log,
                            ih_discount_sum,
                            lc_discount_sum)

# Import SCF Lorenz data from parameters.py
from parameters import (emp_lorenz, MyPopulation, LifeCycle,
                        HetTypeCount, DstnType, DstnParamMapping, BaseTypeCount, tag, TargetPercentiles)

# Specify the center and spread from a previous successful estimation.py run
# You can find these values in the Results directory

#UNIFORM DIST
#PY
#center=1.0183036346087888
#spread=0.07088259593923484


#LC
center=1.0012626349118519
spread=0.09623861461390619


#Lognormal DIST
#PY
#center = 1.027794207449135
#spread = 0.03112001157177155


#LC
#center=1.0094751606448389
#spread=0.04710097282843928



print(f"Using center: {center}, spread: {spread}")
print(f"SCF Lorenz data: {emp_lorenz}")

# Step 1: Extract income distribution and calculate GDP target
# This will update the population with the given center/spread and extract distributions
GDP, WealthDstn, ProdDstn, WeightDstn = extract_income_distribution(center, spread)

# Baseline Lorenz from first-pass arrays to match estimation output
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
print("\nExtracting wealth distributions for taxed populations...")

WealthArray_WT, ProdArray_WT, WeightArray_WT = extract_wealth_distribution(MyPopulation_WT)
WealthArray_CIT, ProdArray_CIT, WeightArray_CIT = extract_wealth_distribution(MyPopulation_CIT)

print(f"Wealth tax population wealth array length: {len(WealthArray_WT)}")
print(f"Capital income tax population wealth array length: {len(WealthArray_CIT)}")

# Summary statistics
print(f"\nOriginal population average wealth (first pass): {np.average(WealthDstn, weights=WeightDstn):.2f}")
print(f"Wealth tax population average wealth: {np.average(WealthArray_WT, weights=WeightArray_WT):.2f}")
print(f"Capital income tax population average wealth: {np.average(WealthArray_CIT, weights=WeightArray_CIT):.2f}")

# Compute Lorenz shares for taxed distributions (baseline from first pass already computed)
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

# Compute S for log-utility case; otherwise use standard CE
if abs(rho - 1.0) < 1e-12:
    # Per-type S from baseline agents (same beta, LivPrb across regimes)
    S_per_type = []
    for i in range(len(W_orig_vec)):
        rep_ag = MyPopulation[i * BaseTypeCount]
        S_per_type.append(ih_discount_sum(rep_ag))

    # Aggregate S by pmv and by realized counts
    import numpy as _np
    S_pmv = float(_np.dot(_np.array(S_per_type), _np.array(het_weights)))

    # Type counts from baseline population
    type_counts = _np.array([
        _np.sum([float(getattr(ag, 'AgentCount', 0.0)) for ag in MyPopulation[i*BaseTypeCount:(i+1)*BaseTypeCount]])
        for i in range(len(W_orig_vec))
    ], dtype=float)
    S_counts = float(_np.dot(_np.array(S_per_type), (type_counts / type_counts.sum())))

    # Aggregate CE comparisons (pmv and counts)
    CE_WT_vs_CIT_pmv     = consumption_equivalent_delta_log(W_wt_pmv,  W_cit_pmv,  S_pmv)
    CE_WT_vs_ORG_pmv     = consumption_equivalent_delta_log(W_wt_pmv,  W_orig_pmv, S_pmv)
    CE_CIT_vs_ORG_pmv    = consumption_equivalent_delta_log(W_cit_pmv, W_orig_pmv, S_pmv)

    CE_WT_vs_CIT_counts  = consumption_equivalent_delta_log(W_wt_counts,  W_cit_counts,  S_counts)
    CE_WT_vs_ORG_counts  = consumption_equivalent_delta_log(W_wt_counts,  W_orig_counts, S_counts)
    CE_CIT_vs_ORG_counts = consumption_equivalent_delta_log(W_cit_counts, W_orig_counts, S_counts)

    # Per-type CE comparisons (WT vs CIT)
    CE_per_type_WT_vs_CIT = [
        consumption_equivalent_delta_log(W_wt_vec[i], W_cit_vec[i], S_per_type[i]) for i in range(len(W_orig_vec))
    ]
else:
    # Aggregate CE comparisons (both weighting modes)
    CE_WT_vs_CIT_pmv    = consumption_equivalent_delta(W_wt_pmv,  W_cit_pmv,  rho)
    CE_WT_vs_ORG_pmv    = consumption_equivalent_delta(W_wt_pmv,  W_orig_pmv, rho)
    CE_CIT_vs_ORG_pmv   = consumption_equivalent_delta(W_cit_pmv, W_orig_pmv, rho)

    CE_WT_vs_CIT_counts = consumption_equivalent_delta(W_wt_counts,  W_cit_counts,  rho)
    CE_WT_vs_ORG_counts = consumption_equivalent_delta(W_wt_counts,  W_orig_counts, rho)
    CE_CIT_vs_ORG_counts= consumption_equivalent_delta(W_cit_counts, W_orig_counts, rho)

    # Per-type CE comparisons (WT vs CIT)
    CE_per_type_WT_vs_CIT = [consumption_equivalent_delta(W_wt_vec[i], W_cit_vec[i], rho) for i in range(len(W_orig_vec))]

# Prepare welfare output file and header BEFORE any LC appends
_welfare_path = os.path.join(_results_dir, f"{tag}_welfare.txt")
_wl = []
_wl.append(f"Tag: {tag}\n")
_wl.append(f"CRRA (rho): {rho:.6f}\n")
_wl.append(f"Wealth tax rate: {wealth_tax_rate:.6f}\n")
_wl.append(f"Capital income tax rate: {capital_income_tax_rate:.6f}\n\n")

# ===================== Life-Cycle specific diagnostics and welfare (if LC) =====================
if LifeCycle:
    print("\n[LC] Education-group diagnostics and per-type returns")
    edu_labels = ["NoHS", "HS", "College"]

    def _collect_type_r_by_edu(pop):
        per_edu = {lab: [] for lab in edu_labels}
        for i in range(HetTypeCount):
            for j, lab in enumerate(edu_labels):
                ag = pop[i * BaseTypeCount + j]
                r = ag.Rfree
                r_val = float(r[0]) if isinstance(r, list) else float(r)
                per_edu[lab].append(r_val)
        return per_edu

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
    print(f"[LC] AgentCount totals by education: {base_counts}")
    print(f"[LC] AgentCount shares by education: {base_shares}")

    base_r_by_edu = _collect_type_r_by_edu(MyPopulation)
    wt_r_by_edu   = _collect_type_r_by_edu(MyPopulation_WT)
    cit_r_by_edu  = _collect_type_r_by_edu(MyPopulation_CIT)

    print("[LC] Baseline returns per education (7 types, low→high):")
    for lab in edu_labels:
        print(f"  {lab}: {base_r_by_edu[lab]}")
    print("[LC] Wealth tax returns per education:")
    for lab in edu_labels:
        print(f"  {lab}: {wt_r_by_edu[lab]}")
    print("[LC] Capital income tax returns per education:")
    for lab in edu_labels:
        print(f"  {lab}: {cit_r_by_edu[lab]}")

    # Diagnostics: parameters used in LC newborn EV (per education)
    print("\n[LC] Newborn EV parameter diagnostics (per education):")
    for j, lab in enumerate(edu_labels):
        rep_ag = MyPopulation[0 * BaseTypeCount + j]
        # Period-0 R and G
        R0 = float(rep_ag.Rfree[0] if isinstance(rep_ag.Rfree, list) else rep_ag.Rfree)
        G0 = float(rep_ag.PermGroFac[0] if isinstance(rep_ag.PermGroFac, list) else rep_ag.PermGroFac)
        # Shocks at t=0
        dist0 = rep_ag.IncShkDstn[0]
        probs0 = getattr(dist0, 'pmv', None)
        atoms0 = getattr(dist0, 'atoms', None)
        if probs0 is not None and atoms0 is not None:
            psi0 = np.asarray(atoms0[0], dtype=float).flatten()
            th0  = np.asarray(atoms0[1], dtype=float).flatten()
            p0   = np.asarray(probs0, dtype=float).flatten()
            p0   = p0 / np.sum(p0)
            # Weighted means for diagnostics
            psi_mean = float(np.dot(psi0, p0))
            th_mean  = float(np.dot(th0,  p0))
        else:
            psi0 = th0 = p0 = None
            psi_mean = th_mean = float('nan')
        # Initial asset distribution (normalized)
        a_init = getattr(rep_ag, 'aNrmInit', None)
        if a_init is not None and hasattr(a_init, 'atoms') and hasattr(a_init, 'pmv'):
            a_vals = np.asarray(a_init.atoms, dtype=float).flatten()
            a_wts  = np.asarray(a_init.pmv,   dtype=float).flatten()
            a_wts  = a_wts / np.sum(a_wts)
            a_mean = float(np.dot(a_vals, a_wts))
            a_min  = float(np.min(a_vals)) if a_vals.size > 0 else float('nan')
            a_max  = float(np.max(a_vals)) if a_vals.size > 0 else float('nan')
            a_len  = int(a_vals.size)
        else:
            a_mean = 0.0
            a_min = a_max = 0.0
            a_len = 1
        print(f"  {lab}: R0={R0:.8f}, G0={G0:.8f}, a0_len={a_len}, a0_mean={a_mean:.8f}, a0_min={a_min:.8f}, a0_max={a_max:.8f}, psi_mean={psi_mean:.8f}, theta_mean={th_mean:.8f}")

    # Additional diagnostics: initial distributions attached to agents
    print("\n[LC] Initial distribution attachments (per education):")
    for j, lab in enumerate(edu_labels):
        ag = MyPopulation[0 * BaseTypeCount + j]
        # aLvlInitDstn and pLvlInitDstn means if present
        try:
            aLvl = getattr(ag, 'aLvlInitDstn', None)
            pLvl = getattr(ag, 'pLvlInitDstn', None)
            if aLvl is not None and hasattr(aLvl, 'atoms') and hasattr(aLvl, 'pmv'):
                a_atoms = np.asarray(aLvl.atoms, dtype=float).flatten()
                a_pmv   = np.asarray(aLvl.pmv,   dtype=float).flatten()
                a_pmv   = a_pmv / np.sum(a_pmv)
                aLvl_mean = float(np.dot(a_atoms, a_pmv))
            else:
                aLvl_mean = float('nan')
            if pLvl is not None and hasattr(pLvl, 'atoms') and hasattr(pLvl, 'pmv'):
                p_atoms = np.asarray(pLvl.atoms, dtype=float).flatten()
                p_pmv   = np.asarray(pLvl.pmv,   dtype=float).flatten()
                p_pmv   = p_pmv / np.sum(p_pmv)
                pLvl_mean = float(np.dot(p_atoms, p_pmv))
            else:
                pLvl_mean = float('nan')
        except Exception:
            aLvl_mean = pLvl_mean = float('nan')

        # aNrmInit mean if present
        try:
            aNrm = getattr(ag, 'aNrmInit', None)
            if aNrm is not None and hasattr(aNrm, 'atoms') and hasattr(aNrm, 'pmv'):
                an_atoms = np.asarray(aNrm.atoms, dtype=float).flatten()
                an_pmv   = np.asarray(aNrm.pmv,   dtype=float).flatten()
                an_pmv   = an_pmv / np.sum(an_pmv)
                aNrm_mean = float(np.dot(an_atoms, an_pmv))
            else:
                aNrm_mean = float('nan')
        except Exception:
            aNrm_mean = float('nan')

        # History means at t=0
        try:
            aLvl0 = np.asarray(ag.history['aLvl'][0, :], dtype=float)
            pLvl0 = np.asarray(ag.history['pLvl'][0, :], dtype=float)
            hist_aLvl_mean = float(np.mean(aLvl0))
            hist_pLvl_mean = float(np.mean(pLvl0))
            hist_aNrm_mean = float(np.mean(aLvl0 / pLvl0))
        except Exception:
            hist_aLvl_mean = hist_pLvl_mean = hist_aNrm_mean = float('nan')

        print(f"  {lab}: aLvlInit_mean={aLvl_mean:.8f}, pLvlInit_mean={pLvl_mean:.8f}, aNrmInit_mean={aNrm_mean:.8f}, hist_aLvl0_mean={hist_aLvl_mean:.8f}, hist_pLvl0_mean={hist_pLvl_mean:.8f}, hist_aNrm0_mean={hist_aNrm_mean:.8f}")

    # Per-education per-type newborn EVs and WT vs CIT CE
    def _evs_by_edu(pop):
        evs = {lab: [] for lab in edu_labels}
        for i in range(HetTypeCount):
            for j, lab in enumerate(edu_labels):
                ag = pop[i * BaseTypeCount + j]
                require_vfunc(ag)
                if LifeCycle:
                    evs[lab].append(compute_newborn_EV_per_agent_lc(ag))
                else:
                    evs[lab].append(compute_newborn_EV_per_agent(ag))
        return evs

    ev_orig_by_edu = _evs_by_edu(MyPopulation)
    ev_wt_by_edu   = _evs_by_edu(MyPopulation_WT)
    ev_cit_by_edu  = _evs_by_edu(MyPopulation_CIT)

    # Per-education S for log utility
    if abs(rho - 1.0) < 1e-12:
        S_by_edu = {}
        for j, lab in enumerate(edu_labels):
            # representative agent of this education at type 0
            rep_ag = MyPopulation[0 * BaseTypeCount + j]
            S_by_edu[lab] = lc_discount_sum(rep_ag)
    else:
        S_by_edu = None

    # Per-education per-type CE (WT vs CIT only)
    CE_by_edu = {lab: [] for lab in edu_labels}
    for j, lab in enumerate(edu_labels):
        for i in range(HetTypeCount):
            if abs(rho - 1.0) < 1e-12:
                CE_by_edu[lab].append(consumption_equivalent_delta_log(ev_wt_by_edu[lab][i], ev_cit_by_edu[lab][i], S_by_edu[lab]))
            else:
                CE_by_edu[lab].append(consumption_equivalent_delta(ev_wt_by_edu[lab][i], ev_cit_by_edu[lab][i], rho))

    # Aggregates in LC across returns (pmv) and education (declared fractions vs counts)
    # Declared fractions
    from parameters import (year)
    nohs_frac = 0.11
    hs_frac = 0.54
    college_frac = 0.35
    edu_fracs = {"NoHS": nohs_frac, "HS": hs_frac, "College": college_frac}

    # Aggregate EVs
    def _agg_ev(evs_by_edu, weights_returns, weights_edu):
        # average across returns per edu, then across edu by weights_edu
        import numpy as _np
        ev_edu = {lab: float(_np.dot(_np.array(evs_by_edu[lab], dtype=float), _np.array(weights_returns, dtype=float))) for lab in edu_labels}
        W_pmv = float(sum(ev_edu[lab] * weights_edu[lab] for lab in edu_labels))
        return ev_edu, W_pmv

    # weights across returns
    ret_weights = list(map(float, dstn.pmv))

    ev_edu_orig, W_orig_pmv_lc = _agg_ev(ev_orig_by_edu, ret_weights, edu_fracs)
    ev_edu_wt,   W_wt_pmv_lc   = _agg_ev(ev_wt_by_edu,   ret_weights, edu_fracs)
    ev_edu_cit,  W_cit_pmv_lc  = _agg_ev(ev_cit_by_edu,  ret_weights, edu_fracs)

    # AgentCount-weighted aggregates
    base_counts, base_shares = _collect_agent_counts_by_edu(MyPopulation)
    W_orig_counts_lc = float(sum(ev_edu_orig[lab] * base_shares[lab] for lab in edu_labels))
    W_wt_counts_lc   = float(sum(ev_edu_wt[lab]   * base_shares[lab] for lab in edu_labels))
    W_cit_counts_lc  = float(sum(ev_edu_cit[lab]  * base_shares[lab] for lab in edu_labels))

    # Aggregate CE (pmv and counts)
    if abs(rho - 1.0) < 1e-12:
        # Aggregate S via same weights
        S_pmv_lc    = float(sum(S_by_edu[lab] * edu_fracs[lab] for lab in edu_labels))
        S_counts_lc = float(sum(S_by_edu[lab] * base_shares[lab] for lab in edu_labels))

        CE_WT_vs_CIT_pmv_lc     = consumption_equivalent_delta_log(W_wt_pmv_lc,  W_cit_pmv_lc,  S_pmv_lc)
        CE_WT_vs_ORG_pmv_lc     = consumption_equivalent_delta_log(W_wt_pmv_lc,  W_orig_pmv_lc, S_pmv_lc)
        CE_CIT_vs_ORG_pmv_lc    = consumption_equivalent_delta_log(W_cit_pmv_lc, W_orig_pmv_lc, S_pmv_lc)

        CE_WT_vs_CIT_counts_lc  = consumption_equivalent_delta_log(W_wt_counts_lc,  W_cit_counts_lc,  S_counts_lc)
        CE_WT_vs_ORG_counts_lc  = consumption_equivalent_delta_log(W_wt_counts_lc,  W_orig_counts_lc, S_counts_lc)
        CE_CIT_vs_ORG_counts_lc = consumption_equivalent_delta_log(W_cit_counts_lc, W_orig_counts_lc, S_counts_lc)
    else:
        CE_WT_vs_CIT_pmv_lc     = consumption_equivalent_delta(W_wt_pmv_lc,  W_cit_pmv_lc,  rho)
        CE_WT_vs_ORG_pmv_lc     = consumption_equivalent_delta(W_wt_pmv_lc,  W_orig_pmv_lc, rho)
        CE_CIT_vs_ORG_pmv_lc    = consumption_equivalent_delta(W_cit_pmv_lc, W_orig_pmv_lc, rho)

        CE_WT_vs_CIT_counts_lc  = consumption_equivalent_delta(W_wt_counts_lc,  W_cit_counts_lc,  rho)
        CE_WT_vs_ORG_counts_lc  = consumption_equivalent_delta(W_wt_counts_lc,  W_orig_counts_lc, rho)
        CE_CIT_vs_ORG_counts_lc = consumption_equivalent_delta(W_cit_counts_lc, W_orig_counts_lc, rho)

    # Append LC results to welfare file
    _wl.append("\n[Life-Cycle] Per-education newborn EVs (Original):\n")
    for lab in edu_labels:
        _wl.append(f"{lab}: {[float(x) for x in np.round(ev_orig_by_edu[lab], 8)]}\n")
    _wl.append("[Life-Cycle] Per-education newborn EVs (Wealth tax):\n")
    for lab in edu_labels:
        _wl.append(f"{lab}: {[float(x) for x in np.round(ev_wt_by_edu[lab], 8)]}\n")
    _wl.append("[Life-Cycle] Per-education newborn EVs (Capital income tax):\n")
    for lab in edu_labels:
        _wl.append(f"{lab}: {[float(x) for x in np.round(ev_cit_by_edu[lab], 8)]}\n")

    _wl.append("\n[Life-Cycle] Per-education CE (WT vs CIT), Δ by return type (low→high):\n")
    for lab in edu_labels:
        _wl.append(f"{lab}: {[float(x) for x in np.round(CE_by_edu[lab], 8)]}\n")

    _wl.append("\n[Life-Cycle] Aggregate newborn welfare (pmv using declared edu fractions):\n")
    _wl.append(f"Original: {W_orig_pmv_lc:.8f}\n")
    _wl.append(f"Wealth tax: {W_wt_pmv_lc:.8f}\n")
    _wl.append(f"Capital income tax: {W_cit_pmv_lc:.8f}\n")

    _wl.append("[Life-Cycle] Aggregate newborn welfare (AgentCount shares):\n")
    _wl.append(f"Original: {W_orig_counts_lc:.8f}\n")
    _wl.append(f"Wealth tax: {W_wt_counts_lc:.8f}\n")
    _wl.append(f"Capital income tax: {W_cit_counts_lc:.8f}\n")

    _wl.append("\n[Life-Cycle] Aggregate CE (pmv), Δ:\n")
    _wl.append(f"WT vs CIT: {CE_WT_vs_CIT_pmv_lc:.8f}\n")
    _wl.append(f"WT vs Original: {CE_WT_vs_ORG_pmv_lc:.8f}\n")
    _wl.append(f"CIT vs Original: {CE_CIT_vs_ORG_pmv_lc:.8f}\n")

    _wl.append("[Life-Cycle] Aggregate CE (AgentCount shares), Δ:\n")
    _wl.append(f"WT vs CIT: {CE_WT_vs_CIT_counts_lc:.8f}\n")
    _wl.append(f"WT vs Original: {CE_WT_vs_ORG_counts_lc:.8f}\n")
    _wl.append(f"CIT vs Original: {CE_CIT_vs_ORG_counts_lc:.8f}\n")

# Save welfare results to Results_taxes/{tag}_welfare.txt
# For PY (IH), append standard aggregates; for LC, skip (LC-specific section already written above)
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

    # Also print baseline and taxed returns per type for interpretation
    try:
        _wl.append("\nPer-type returns (baseline, WT, CIT) by return type (low→high):\n")
        # baseline returns from discretization atoms
        baseline_R = list(map(float, interest_rates))
        # taxed returns per type: apply the same policy rules used in helpers
        def _wt_print(r):
            return float(r - wealth_tax_rate)
        def _cit_print(r):
            r = float(r)
            if r > 1.0:
                return float(1.0 + (r - 1.0) * (1.0 - capital_income_tax_rate))
            else:
                return r
        wt_R = [_wt_print(r) for r in baseline_R]
        cit_R = [_cit_print(r) for r in baseline_R]
        _wl.append(f"Baseline R: {baseline_R}\n")
        _wl.append(f"Wealth tax R: {wt_R}\n")
        _wl.append(f"Capital income tax R: {cit_R}\n")
    except Exception:
        pass

with open(_welfare_path, 'w', encoding='utf-8') as _wf:
    _wf.writelines(_wl)
print(f"Saved newborn welfare results to: {_welfare_path}")
