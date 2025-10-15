# Custom Value Functions and Welfare Analysis for Perpetual Youth

## Phase 0: Clean Slate - Remove All Existing Welfare Code

### Remove welfare computation from `Code/taxes_compute.py`

Starting at line 350 ("Newborn welfare computation and outputs"), remove all code related to:

- `require_vfunc` calls and validation
- `compute_newborn_EV_per_agent` calls
- `compute_population_newborn_welfare` calls
- All `W_orig_vec`, `W_wt_vec`, `W_cit_vec` variables
- All aggregate welfare computations (`W_orig_pmv`, `W_wt_pmv`, etc.)
- All CE calculations (`consumption_equivalent_delta`, `consumption_equivalent_delta_log`)
- All `S` discount sum calculations (`ih_discount_sum`, `lc_discount_sum`)
- The entire welfare output file writing section (lines ~424-741)
- Keep only: tax rate computation, population creation/solving, wealth distribution extraction, and Lorenz curve outputs

### Remove welfare functions from `Code/utilities_taxes.py`

Remove these functions entirely:

- `require_vfunc()`
- `compute_newborn_EV_per_agent()`
- `compute_newborn_EV_per_agent_lc()`
- `compute_population_newborn_welfare()`
- `consumption_equivalent_delta()`
- `consumption_equivalent_delta_log()`
- `ih_discount_sum()`
- `lc_discount_sum()`

Keep only:

- `getDistributionsFromHetParamValues()`
- `extract_income_distribution()`
- `make_Rfree_with_wealth_tax()`
- `make_Rfree_with_capital_income_tax()`
- `save_lorenz_side_by_side_from_results()`

### Verify clean baseline

After cleanup, `taxes_compute.py` should:

1. Load center/spread from estimation
2. Compute revenue-equivalent tax rates
3. Create taxed populations and solve them
4. Extract wealth distributions
5. Compute and save Lorenz curves
6. Export results to `Results/Results_taxes/{tag}.txt` (NO welfare file)

## Phase 1: Build Custom Value Function Infrastructure

### Create new module `Code/value_functions.py`

This module will contain all custom value function computation logic.

#### Core function: `compute_value_function_bellman(agent, m_grid, max_iter=100, tol=1e-6)`

Computes V(m) via Bellman iteration for a solved agent.

**Inputs:**

- `agent`: Solved HARK agent with consumption function `agent.solution[0].cFunc`
- `m_grid`: Array of m values to compute V(m) on (e.g., `np.linspace(0.01, 20, 200)`)
- `max_iter`: Maximum iterations for convergence
- `tol`: Convergence tolerance

**Algorithm:**

1. Extract parameters from agent:

   - `c_func = agent.solution[0].cFunc`
   - `beta = agent.DiscFac`
   - `R = agent.Rfree[0]` (or scalar)
   - `Gamma = agent.PermGroFac[0]` (or scalar)
   - `rho = agent.CRRA`
   - `dist = agent.IncShkDstn[0]` (shock distribution)

2. Define utility function:
   ```python
   if abs(rho - 1.0) < 1e-12:
       u = lambda c: np.log(c)
   else:
       u = lambda c: (c**(1-rho)) / (1-rho)
   ```

3. Initialize: `V_old = np.zeros(len(m_grid))`

4. Iterate until convergence:

   - For each m in m_grid:
     - Compute `c = c_func(m)`
     - Compute `a = m - c`
     - Compute expected continuation value:
       ```python
       EV_next = 0.0
       for prob, psi, theta in zip(dist.pmv, dist.atoms[0], dist.atoms[1]):
           m_next = (R / (Gamma * psi)) * a + theta
           V_next = np.interp(m_next, m_grid, V_old)  # Linear interpolation
           EV_next += prob * V_next
       ```

     - Update: `V_new[i] = u(c) + beta * EV_next`
   - Check convergence: `max(abs(V_new - V_old)) < tol`
   - Update: `V_old = V_new`

5. Return: `(m_grid, V_grid)` and convergence info

#### Helper function: `create_value_function_interpolator(m_grid, V_grid)`

Creates an interpolator from grid values.

**Returns:** Function `V_func(m)` using scipy's `interp1d` (linear, with extrapolation handling)

## Phase 2: Compute Newborn Expected Values

### Add to `Code/value_functions.py`:

#### Function: `compute_newborn_EV_PY(agent, V_func)`

Computes expected lifetime utility for a PY newborn.

**Algorithm:**

1. Get shock distribution: `dist = agent.IncShkDstn[0]`
2. For PY newborns, `a_nrm = 0`, so `m = theta`
3. Integrate:
   ```python
   EV = 0.0
   for prob, theta in zip(dist.pmv, dist.atoms[1]):
       m_newborn = float(theta)
       EV += float(prob) * V_func(m_newborn)
   return EV
   ```


#### Function: `compute_population_welfare_custom(population, het_weights, base_type_count)`

Computes per-type EVs and aggregate welfare.

**Algorithm:**

1. For each return type `i` (HetTypeCount types total):

   - Extract agents for this type: `block = population[i*base_type_count : (i+1)*base_type_count]`
   - For each agent in block:
     - Compute V(m) using `compute_value_function_bellman()`
     - Create interpolator `V_func`
     - Compute `EV = compute_newborn_EV_PY(agent, V_func)`
   - Average EVs across base types: `W_vec[i] = mean(EVs)`

2. Aggregate welfare (pmv-weighted):
   ```python
   W_avg_pmv = np.dot(W_vec, het_weights)
   ```

3. Aggregate welfare (AgentCount-weighted):
   ```python
   type_counts = [sum agent counts per type]
   W_avg_counts = np.dot(W_vec, type_counts / total_counts)
   ```


**Returns:** `(W_vec, W_avg_pmv, W_avg_counts)`

## Phase 3: Consumption-Equivalent Welfare Comparisons

### Add to `Code/value_functions.py`:

#### Function: `ih_discount_sum_custom(agent)`

Computes S = sum_{t>=0} (beta * LivPrb)^t for infinite horizon.

**Algorithm:**

```python
beta = agent.DiscFac (as scalar)
liv = agent.LivPrb[0] (as scalar)
eff = beta * liv
if eff >= 1.0:
    raise ValueError("beta*LivPrb >= 1; discount sum diverges")
S = 1.0 / (1.0 - eff)
return S
```

#### Function: `consumption_equivalent_delta_custom(W_A, W_B, rho, S=None)`

Computes CE welfare difference Δ such that agent in A needs (1+Δ) consumption to match B.

**Algorithm:**

```python
if abs(rho - 1.0) < 1e-12:  # Log utility
    if S is None:
        raise ValueError("Must provide S for log utility")
    delta = exp((W_B - W_A) / S) - 1.0
else:  # CRRA
    delta = (W_B / W_A)**(1.0 / (1.0 - rho)) - 1.0
return delta
```

## Phase 4: Integration into Tax Analysis

### Modify `Code/taxes_compute.py`:

After line ~348 (after Lorenz output), add new welfare computation section:

```python
# ===== Custom Value Function Welfare Analysis =====
from value_functions import (
    compute_value_function_bellman,
    create_value_function_interpolator,
    compute_population_welfare_custom,
    ih_discount_sum_custom,
    consumption_equivalent_delta_custom
)

# Set up m_grid for value function computation
m_grid = np.linspace(0.01, 20.0, 200)

# Compute welfare for each regime
print("\nComputing welfare using custom value functions...")

# Get heterogeneity weights from discretization
het_weights = dstn.pmv

# Original (no tax)
W_orig_vec, W_orig_pmv, W_orig_counts = compute_population_welfare_custom(
    MyPopulation, het_weights, BaseTypeCount, m_grid
)

# Wealth tax
W_wt_vec, W_wt_pmv, W_wt_counts = compute_population_welfare_custom(
    MyPopulation_WT, het_weights, BaseTypeCount, m_grid
)

# Capital income tax
W_cit_vec, W_cit_pmv, W_cit_counts = compute_population_welfare_custom(
    MyPopulation_CIT, het_weights, BaseTypeCount, m_grid
)

# Get CRRA and compute S for log utility
rho = float(MyPopulation[0].CRRA)

if abs(rho - 1.0) < 1e-12:
    # Per-type S values
    S_per_type = [ih_discount_sum_custom(MyPopulation[i * BaseTypeCount])
                  for i in range(len(W_orig_vec))]

    # Aggregate S (pmv and counts weighted)
    S_pmv = np.dot(S_per_type, het_weights)
    type_counts = [sum agent counts per type]
    S_counts = np.dot(S_per_type, type_counts / total_counts)

    # Aggregate CE comparisons
    CE_WT_vs_CIT_pmv = consumption_equivalent_delta_custom(W_wt_pmv, W_cit_pmv, rho, S_pmv)
    CE_WT_vs_ORG_pmv = consumption_equivalent_delta_custom(W_wt_pmv, W_orig_pmv, rho, S_pmv)
    CE_CIT_vs_ORG_pmv = consumption_equivalent_delta_custom(W_cit_pmv, W_orig_pmv, rho, S_pmv)

    # (similar for counts)

    # Per-type CE comparisons
    CE_per_type_WT_vs_CIT = [
        consumption_equivalent_delta_custom(W_wt_vec[i], W_cit_vec[i], rho, S_per_type[i])
        for i in range(len(W_orig_vec))
    ]
else:
    # Non-log utility CE (no S needed)
    # ... similar structure without S parameter
```

### Create welfare output file

Write results to `Results/Results_taxes/{tag}_welfare.txt`:

- Tag and parameters (CRRA, tax rates)
- Per-type newborn EVs (Original, WT, CIT)
- Aggregate welfare (pmv and counts weighted)
- Aggregate CE comparisons (WT vs CIT, WT vs Original, CIT vs Original)
- Per-type CE comparisons (WT vs CIT)
- Per-type returns (baseline, WT, CIT) for interpretation

## Phase 5: Testing and Validation

### Test with existing PY specification

Run `taxes_compute.py` with:

- Uniform distribution PY center/spread
- Verify outputs match economic intuition:
  - Low return types should prefer capital income tax
  - High return types should prefer wealth tax
  - Preferences should be monotonic in return type

### Diagnostics to add:

- Print convergence info from Bellman iteration
- Print min/max values of V(m) on grid
- Print per-type S values for log utility
- Verify CE signs align with post-tax return rankings

## Success Criteria

1. Clean baseline: `taxes_compute.py` runs without welfare code
2. Custom value functions converge for all agents
3. Newborn EVs computed successfully for all return types
4. CE welfare comparisons are economically consistent
5. Results written to welfare file with clear interpretation
6. All diagnostics pass

## Notes

- This plan focuses on PY only; LC extension will be separate
- All code uses first principles (Bellman equation) - no HARK vFunc dependencies
- Value function computation is transparent and auditable
- Ready for LC extension once PY is validated

### To-dos

- [ ] Remove all existing welfare code from taxes_compute.py and utilities_taxes.py
- [ ] Create value_functions.py with Bellman iteration infrastructure
- [ ] Add newborn EV computation functions for PY
- [ ] Add consumption-equivalent welfare comparison functions
- [ ] Integrate custom value functions into taxes_compute.py
- [ ] Test and validate with PY specification
