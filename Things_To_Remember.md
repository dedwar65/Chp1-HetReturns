# Things to Remember - User Requests

## Key Points User Asked Me to Remember

### 1. CRRA=1 Numerical Issues
- **User explicitly requested** to revert to the original approach where `CRRA=1.0` was kept, but a small nudge (e.g., `CRRA=1.01`) was applied *only* in the `vNvrsP` calculation (derivative of inverse value function) when `vFuncBool=True`
- This was to avoid numerical issues while maintaining the economic interpretation of log utility
- **User rejected** the approach of using `CRRA=1.001` as a workaround, insisting on making `CRRA=1.0` work

### 2. Economic Constraint for Log Utility
- For infinite horizon and log utility, `β * Rfree < 1` is required for the value function to be finite
- This is a fundamental economic constraint that cannot be violated

### 3. Tax Application Logic
- **Capital Income Tax**: Only taxes positive capital income (R > 1)
- **Wealth Tax**: Taxes all wealth uniformly
- The defining property: when someone earns no capital income, they essentially avoid the capital income tax, but they can't avoid the wealth tax

### 4. Revenue Target Scaling
- **User's intuition**: Higher revenue targets (e.g., 5% vs 1% of GDP) should make it less likely for high types to prefer wealth tax over capital income tax
- **Mathematical analysis showed**: Both tax rates scale by the same factor, so relative preferences don't change, only magnitudes

### 5. Non-Monotonic Welfare Pattern
- **User's concern**: The pattern where 3rd highest and highest return types prefer wealth tax, but 2nd highest prefers capital income tax is "strange" and "difficult to justify to a crowd"
- **Critical issue identified**: Type 5 prefers wealth tax (CE = -0.0004765) despite having higher post-tax return under capital income tax (1.0371 > 1.0357)
- This is economically impossible and needs to be debugged

### 6. Expected Value Computation
- **User's main concern**: "My issue has always been the EV. Don't really understand how the expected value is being computed"
- Need to understand every mathematical equation and its mapping into the HARK toolkit
- The EV computation depends on: value function, initial conditions, expectation over initial shocks

### 7. Buffer Stock Theory References
- HARK and its solution methods are built around Carroll's Buffer Stock Theory paper
- Also referenced Carroll's endogenous grid method paper
- These provide the mathematical foundation for understanding the value function and welfare calculations

### 8. Debugging Priority
- **User's request**: "we need to debug and just make sure the computations are correct"
- The non-monotonic pattern suggests fundamental issues with either tax implementation or welfare calculation
- Cannot present results until this inconsistency is resolved

### 9. Mathematical Foundation
- Value function: `V(m) = max_c {u(c) + β E[V(m')]}`
- In HARK: `vFunc(m) = CRRAutility(vFuncNvrs(m), CRRA)`
- For CRRA=1: `vFunc(m) = log(vFuncNvrs(m))`
- Expected value: `EV = E[V(m_0)]` where `m_0 = (a_0 + y_0)/p_0`
- Consumption-equivalent: `CE = exp((W_CIT - W_WT)/S) - 1`

### 10. Current Tax Rates (1% GDP target)
- Wealth Tax Rate: 0.003407 (0.3407%)
- Capital Income Tax Rate: 0.051097 (5.1097%)
- For 5% target: Wealth tax becomes 1.672%, Capital income tax becomes 24.875%

## Files and Results to Remember
- Results from `Lognorm_PYrrDistNetWorth_2004_welfare.txt` show the problematic non-monotonic pattern
- The `S` (discount sum) calculation for log utility needs verification
- Tax application to value functions vs. just returns needs to be checked
