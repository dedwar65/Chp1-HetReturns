# Welfare Analysis Debug Notes

## Overview
Analysis of welfare effects of different tax schemes (wealth tax vs. capital income tax) on "newborn" agents in HARK-based macroeconomic model. The analysis compares expected lifetime utility (EV) and consumption-equivalent (CE) welfare changes under three scenarios: original (no tax change), wealth tax, and capital income tax.

## Key Files and Structure
- **`Code/parameters.py`**: Model parameters, empirical targets, agent population structure
- **`Code/estimation.py`**: Structural estimation, HARK agent solving and simulation logic
- **`Code/utilities.py`**: `AltIndShockConsumerType` subclass and utility functions
- **`Code/utilities_taxes.py`**: Tax application, Lorenz curve plotting, newborn welfare calculations
- **`Code/taxes_compute.py`**: Main script for applying taxes, re-simulating, and computing welfare comparisons

## Current Status
- **Perpetual Youth (PY) Model**: Implemented with 7 return types
- **Life-Cycle (LC) Model**: Extended to 21 types (3 education × 7 return types)
- **CRRA=1 Issues**: Resolved with small nudge (1.01) in `vNvrsP` calculation when `vFuncBool=True`

## Critical Issue: Non-Monotonic Welfare Results

### The Problem
From Lognormal PY results:
```
Per-type consumption-equivalent WT vs CIT, Δ by return type (low→high):
[-0.45381652, 0.02263904, 0.01854466, 0.19803414, -0.0004765, 0.00423176, -0.00376901]

Per-type returns (baseline, WT, CIT) by return type (low→high):
Baseline R: [0.978111409220195, 1.0020523151917913, 1.0155931780043574, 1.0273019916222637, 1.039147389862228, 1.0531979732510361, 1.079155194992074]
Wealth tax R: [0.9747043152411433, 0.9986452212127396, 1.0121860840253059, 1.023894897643212, 1.0357402958831765, 1.0497908792719846, 1.0757481010130223]
Capital income tax R: [0.978111409220195, 1.001947447198333, 1.0147964069745006, 1.025906930527217, 1.037147059581382, 1.0504796945318626, 1.0751105694375693]
```

### The Inconsistency
**Type 5 (R=1.039):**
- Post-tax returns: R_CIT > R_WT (1.0371 > 1.0357)
- CE preference: Prefers Wealth Tax (CE = -0.0004765)
- **This is economically impossible!**

**Type 6 (R=1.053):**
- Post-tax returns: R_CIT > R_WT (1.0505 > 1.0498)
- CE preference: Prefers Capital Income Tax (CE = 0.00423176)
- **This is consistent**

**Type 7 (R=1.079):**
- Post-tax returns: R_WT > R_CIT (1.0757 > 1.0751)
- CE preference: Prefers Wealth Tax (CE = -0.00376901)
- **This is consistent**

## Mathematical Foundation

### Value Function Structure
From Carroll's Buffer Stock Theory:
```
V(m) = max_c {u(c) + β E[V(m')]}
```

In HARK:
```python
vFunc(m) = CRRAutility(vFuncNvrs(m), CRRA)
```

For CRRA=1 (log utility):
```python
vFunc(m) = log(vFuncNvrs(m))
```

### Expected Value for Newborns
```
EV = E[V(m_0)]
```

Where `m_0 = (a_0 + y_0)/p_0` is the initial normalized cash-on-hand.

### Consumption-Equivalent Calculation
```
CE = exp((W_CIT - W_WT)/S) - 1
```

Where:
- `W_CIT` = expected value under capital income tax
- `W_WT` = expected value under wealth tax
- `S` = sum of discounted survival probabilities

## Tax Implementation

### Current Tax Rates (1% GDP target)
- **Wealth Tax Rate**: 0.003407 (0.3407%)
- **Capital Income Tax Rate**: 0.051097 (5.1097%)

### Tax Application
**Wealth Tax:**
```
R_WT = R - τ_w
```

**Capital Income Tax:**
```
R_CIT = 1 + (R-1) × (1-τ_c)  if R > 1
R_CIT = R                    if R ≤ 1
```

## Debugging Questions

1. **How is `m_0` computed** for each return type?
2. **How does the tax affect the value function** computation?
3. **Are we correctly computing** the expectation over initial conditions?
4. **Is the `S` calculation correct** for the discounting?
5. **Are the tax rates being applied correctly** to the value function vs. just the returns?

## Revenue Target Analysis

### 5% GDP Target Calculations
- **New Wealth Tax Rate**: 0.01672 (1.672%) - 5x increase
- **New Capital Income Tax Rate**: 0.24875 (24.875%) - 5x increase

### Key Insight
The relative preferences don't change because both tax rates scale by the same factor, but the magnitude of welfare differences will be much larger.

## Next Steps

1. **Debug the expected value computation** in HARK
2. **Verify tax application** to value functions
3. **Check numerical precision** in CE calculations
4. **Resolve the non-monotonic pattern** before presenting results
5. **Test with 5% revenue target** once debugging is complete

## Files to Examine
- `Code/utilities_taxes.py`: Newborn welfare calculation functions
- `Code/taxes_compute.py`: Main welfare computation logic
- HARK's value function computation methods
- The `S` (discount sum) calculation for log utility

## Economic Constraint
For CRRA=1 (log utility) in infinite horizon: `β * Rfree < 1` is required for finite value function.

## Results Files
- `Results/Results_taxes/Lognorm_PYrrDistNetWorth_2004_welfare.txt`
- `Results/Results_taxes/Unif_PYrrDistNetWorth_2004_welfare.txt`
- `Results/Results_taxes/Lognorm_LCrrDistNetWorth_2004_welfare.txt`
- `Results/Results_taxes/Unif_LCrrDistNetWorth_2004_welfare.txt`
