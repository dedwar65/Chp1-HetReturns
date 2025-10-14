import numpy as np
import matplotlib.pyplot as plt
from utilities import get_lorenz_shares

def getDistributionsFromHetParamValues(center, spread):
    """
    Import the working function from estimation.py instead of recreating it.
    """
    from estimation import getDistributionsFromHetParamValues as estimation_getDistributions
    return estimation_getDistributions(center, spread)

def extract_income_distribution(center, spread):
    """
    Extract income and wealth distributions from the population.
    """
    WealthDstn, ProdDstn, WeightDstn, MPCDstn = getDistributionsFromHetParamValues(center, spread)
    GDP = np.dot(ProdDstn, WeightDstn)
    return GDP, WealthDstn, ProdDstn, WeightDstn

def make_Rfree_with_wealth_tax(RfreeFull, WealthTaxRate, T_cycle):
    """
    Create Rfree array with wealth tax applied.
    """
    if isinstance(RfreeFull, list):
        Rfree = RfreeFull[0]  # Get scalar value from list
    else:
        Rfree = RfreeFull

    # Apply wealth tax: Rfree_new = Rfree - WealthTaxRate
    Rfree_new = Rfree - WealthTaxRate

    # Return as list for T_cycle compatibility
    return [Rfree_new] * T_cycle

def make_Rfree_with_capital_income_tax(RfreeFull, CapitalTaxRate, T_cycle):
    """
    Create Rfree array with capital income tax applied.
    """
    # Element-wise policy: tax only positive capital income; do not subsidize when Rfree<=1
    def _cit_transform(r):
        r = float(r)
        if r > 1.0:
            return 1.0 + (r - 1.0) * (1.0 - CapitalTaxRate)
        else:
            return r

    if isinstance(RfreeFull, list):
        # If list has a single scalar, broadcast; otherwise transform element-wise
        if len(RfreeFull) == 1:
            R_val = float(RfreeFull[0])
            R_new = _cit_transform(R_val)
            return [R_new] * T_cycle
        else:
            R_list = [ _cit_transform(x) for x in RfreeFull ]
            return R_list
    else:
        R_val = float(RfreeFull)
        R_new = _cit_transform(R_val)
        return [R_new] * T_cycle

def save_lorenz_side_by_side_from_results(results_dict, tag, percentiles):
    """
    Create side-by-side Lorenz curve plots for wealth tax and capital income tax.
    Now includes SCF data for comparison.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Get Lorenz data from results_dict
    SCF_lorenz = results_dict['SCF_lorenz']
    original_lorenz = results_dict['original']['lorenz']
    wealth_tax_lorenz = results_dict['wealth_tax']['lorenz']
    capital_income_tax_lorenz = results_dict['capital_income_tax']['lorenz']

    # Ensure arrays are numpy arrays
    percentiles = np.asarray(percentiles, dtype=float)
    SCF_lorenz = np.asarray(SCF_lorenz, dtype=float)
    original_lorenz = np.asarray(original_lorenz, dtype=float)
    wealth_tax_lorenz = np.asarray(wealth_tax_lorenz, dtype=float)
    capital_income_tax_lorenz = np.asarray(capital_income_tax_lorenz, dtype=float)

    # Augment with endpoints to span [0,1]
    x_plot = np.concatenate(([0.0], percentiles, [1.0]))
    SCF_plot = np.concatenate(([0.0], SCF_lorenz, [1.0]))
    orig_plot = np.concatenate(([0.0], original_lorenz, [1.0]))
    wt_plot = np.concatenate(([0.0], wealth_tax_lorenz, [1.0]))
    cit_plot = np.concatenate(([0.0], capital_income_tax_lorenz, [1.0]))

    # Left plot: Wealth Tax
    ax1.plot(x_plot, SCF_plot, 'k-', linewidth=2, label='SCF')
    ax1.plot(x_plot, orig_plot, 'b--', linewidth=2, label='Model (Original)')
    ax1.plot(x_plot, wt_plot, 'r:', linewidth=2, label='Model (Wealth tax)')
    ax1.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='45 Degree')
    ax1.set_xlabel('Percentile of net worth')
    ax1.set_ylabel('Cumulative share of wealth')
    ax1.set_title('Wealth Tax')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc=2)

    # Right plot: Capital Income Tax
    ax2.plot(x_plot, SCF_plot, 'k-', linewidth=2, label='SCF')
    ax2.plot(x_plot, orig_plot, 'b--', linewidth=2, label='Model (Original)')
    ax2.plot(x_plot, cit_plot, 'g:', linewidth=2, label='Model (Capital income tax)')
    ax2.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='45 Degree')
    ax2.set_xlabel('Percentile of net worth')
    ax2.set_ylabel('Cumulative share of wealth')
    ax2.set_title('Capital Income Tax')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc=2)

    # Save the figure
    import os
    os.makedirs('Figures/Figures_taxes', exist_ok=True)
    filename = f'Figures/Figures_taxes/{tag}_lorenz_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Lorenz comparison plot saved to {filename}")
    return filename


# ==== Newborn welfare helpers (Perpetual Youth first; LC-ready aggregation) ====

def require_vfunc(agent):
    """
    Ensure the agent has a value function available (requires vFuncBool=True during solve).
    Raises a RuntimeError with guidance if missing.
    """
    if not hasattr(agent, 'solution') or agent.solution is None or len(agent.solution) == 0:
        raise RuntimeError("Agent has no solution. Ensure vFuncBool=True in YAML and solve() was called.")
    sol0 = agent.solution[0]
    if not hasattr(sol0, 'vFunc') or sol0.vFunc is None:
        raise RuntimeError("Value function missing. Set vFuncBool=True in YAML and re-solve agents.")


def compute_newborn_EV_per_agent(agent):
    """
    Expected lifetime utility for a newborn agent in Perpetual Youth (IH) model.
    Evaluates V(m) at m=theta for the initial period and integrates over income shocks.
    Assumes a_nrm=0 at birth; theta is the transitory shock.
    """
    require_vfunc(agent)

    # Distribution of income shocks in the first period
    dist0 = agent.IncShkDstn[0]
    # pmv: probabilities; atoms: [perm_shocks, tran_shocks]
    probs = getattr(dist0, 'pmv', None)
    atoms = getattr(dist0, 'atoms', None)
    if probs is None or atoms is None:
        raise RuntimeError("IncShkDstn[0] missing pmv/atoms. Check HARK version or distribution API.")

    tran_shocks = atoms[1]
    vFunc = agent.solution[0].vFunc

    EV = 0.0
    for p, theta in zip(probs, tran_shocks):
        # Newborn has a_nrm = 0 => m = theta (normalized)
        EV += float(p) * float(vFunc(float(theta)))
    return float(EV)


def compute_newborn_EV_per_agent_lc(agent):
    """
    Expected lifetime utility for a *life-cycle* newborn at entry (t=0),
    integrating over the SAME initial asset distribution and first-period
    income shocks used by HARK's simulation.

    Uses m0 = (Rfree0/(PermGroFac0 * psi0)) * aNrmInit + theta0,
    and evaluates V_0(m0) with agent.solution[0].vFunc.
    """
    require_vfunc(agent)

    # First-period income shock distribution (education-dependent)
    dist0 = agent.IncShkDstn[0]
    probs = getattr(dist0, 'pmv', None)
    atoms = getattr(dist0, 'atoms', None)
    if probs is None or atoms is None:
        raise RuntimeError("IncShkDstn[0] missing pmv/atoms for LC EV.")
    perm_shocks = atoms[0]
    tran_shocks = atoms[1]

    # Use EXACT initial draws from the simulation at entry (t=0) if available
    a_vals = None
    a_wts = None
    try:
        aLvl0 = agent.history['aLvl'][0, :]
        pLvl0 = agent.history['pLvl'][0, :]
        # Normalize to aNrm at entry
        a_vals_hist = (np.asarray(aLvl0, dtype=float) / np.asarray(pLvl0, dtype=float)).flatten()
        # Use unit weights (or WeightFac at t=0 if available)
        if 'WeightFac' in agent.history:
            w0 = np.asarray(agent.history['WeightFac'][0, :], dtype=float).flatten()
            if np.sum(w0) > 0:
                a_wts_hist = w0 / np.sum(w0)
            else:
                a_wts_hist = np.ones_like(a_vals_hist) / a_vals_hist.size
        else:
            a_wts_hist = np.ones_like(a_vals_hist) / a_vals_hist.size
        a_vals = a_vals_hist
        a_wts = a_wts_hist
    except Exception:
        # Fallback: Initial normalized asset distribution at entry (aNrmInit)
        a_init = getattr(agent, 'aNrmInit', None)
        if a_init is None:
            a_vals = np.array([0.0], dtype=float)
            a_wts = np.array([1.0], dtype=float)
        else:
            a_pmv = getattr(a_init, 'pmv', None)
            a_atoms = getattr(a_init, 'atoms', None)
            if a_pmv is None or a_atoms is None:
                try:
                    a_vals = np.asarray(a_init, dtype=float).flatten()
                    a_wts = np.ones_like(a_vals) / a_vals.size
                except Exception:
                    a_vals = np.array([0.0], dtype=float)
                    a_wts = np.array([1.0], dtype=float)
            else:
                a_vals = np.asarray(a_atoms).flatten().astype(float)
                a_wts = np.asarray(a_pmv, dtype=float).flatten()
                a_wts = a_wts / np.sum(a_wts)

    # Period-0 parameters
    if isinstance(agent.Rfree, list):
        R0 = float(agent.Rfree[0])
    else:
        R0 = float(agent.Rfree)

    if isinstance(agent.PermGroFac, list):
        G0 = float(agent.PermGroFac[0])
    else:
        G0 = float(agent.PermGroFac)

    v0 = agent.solution[0].vFunc

    # Compute EV over (a0, shocks)
    EV = 0.0
    for a_weight, a0 in zip(a_wts, a_vals):
        # For each income shock draw
        for pi, psi, theta in zip(probs, perm_shocks, tran_shocks):
            R_eff = R0 / (G0 * float(psi))
            m0 = R_eff * float(a0) + float(theta)
            EV += float(a_weight) * float(pi) * float(v0(float(m0)))

    return float(EV)

def compute_population_newborn_welfare(population, het_weights, base_type_count):
    """
    Compute newborn EV by return type (W_vec) and aggregate two ways:
    - pmv-weighted using discretization weights (W_avg_pmv)
    - realized AgentCount shares (W_avg_counts)

    Grouping: agents are ordered by return type blocks, each block of size base_type_count.
    """
    if base_type_count <= 0:
        raise ValueError("base_type_count must be positive")

    num_agents = len(population)
    if num_agents % base_type_count != 0:
        raise ValueError("Population size is not a multiple of base_type_count; cannot group by return type.")

    num_types = num_agents // base_type_count

    # Per-type EVs and counts
    W_vec = np.zeros(num_types, dtype=float)
    type_counts = np.zeros(num_types, dtype=float)

    for t in range(num_types):
        start = t * base_type_count
        end = start + base_type_count
        block = population[start:end]

        # Average EV across base types within the same return type
        evs = []
        agent_counts_in_block = []
        for ag in block:
            evs.append(compute_newborn_EV_per_agent(ag))
            agent_counts_in_block.append(float(getattr(ag, 'AgentCount', 0.0)))

        W_vec[t] = float(np.mean(evs)) if len(evs) > 0 else np.nan
        type_counts[t] = float(np.sum(agent_counts_in_block))

    # pmv-weighted aggregate (exact ex-ante weights from discretization)
    if het_weights is None or len(het_weights) != num_types:
        raise ValueError("het_weights must be provided and match number of return types.")
    het_weights = np.asarray(het_weights, dtype=float)
    W_avg_pmv = float(np.dot(W_vec, het_weights))

    # realized AgentCount-weighted aggregate
    total_agents = float(np.sum(type_counts))
    if total_agents > 0:
        W_avg_counts = float(np.dot(W_vec, type_counts / total_agents))
    else:
        W_avg_counts = float('nan')

    return W_vec, W_avg_pmv, W_avg_counts


def consumption_equivalent_delta(W_A, W_B, rho):
    """
    Consumption-equivalent change Δ such that V_A((1+Δ)c) = V_B(c).
    Returns Δ (e.g., 0.07 means +7%).
    """
    W_A = float(W_A)
    W_B = float(W_B)
    rho = float(rho)
    if rho == 1.0:
        raise ValueError(
            "For rho==1 (log utility), use consumption_equivalent_delta_log with S=sum_t beta^t (survival-weighted)."
        )
    # For CRRA (rho!=1): 1+Δ = (W_B/W_A)^{1/(1-ρ)}
    return (W_B / W_A) ** (1.0 / (1.0 - rho)) - 1.0


def consumption_equivalent_delta_log(W_A, W_B, S):
    """
    Consumption-equivalent change Δ for log utility (rho==1):
    1+Δ = exp((W_A - W_B)/S), where S = sum_t beta^t (survival-weighted).
    """
    from math import exp
    W_A = float(W_A)
    W_B = float(W_B)
    S = float(S)
    if S <= 0.0:
        raise ValueError("S must be positive for log-utility CE computation.")
    # For log utility: 1+Δ = exp((W_B - W_A)/S)
    return exp((W_B - W_A) / S) - 1.0


def ih_discount_sum(agent):
    """
    For infinite-horizon (perpetual youth) with constant survival probability,
    returns S = sum_{t>=0} (beta * LivPrb)^t = 1 / (1 - beta*LivPrb).

    Uses the agent's scalar beta (DiscFac) and first-period LivPrb.
    """
    beta = agent.DiscFac
    if isinstance(beta, list):
        beta = float(beta[0])
    else:
        beta = float(beta)

    liv = agent.LivPrb
    if isinstance(liv, list):
        liv = float(liv[0])
    else:
        liv = float(liv)

    eff = beta * liv
    if eff >= 1.0:
        raise ValueError(f"beta*LivPrb >= 1 ({eff}); discount sum diverges.")
    return 1.0 / (1.0 - eff)


def lc_discount_sum(agent):
    """
    Life-cycle discount-sum S_e for log-utility CE comparisons.
    Uses time-varying beta (DiscFac list or scalar) and survival probabilities (LivPrb list).

    S_e = sum_{t=0}^{T-1} (prod_{k=0}^{t-1} beta_k) * CumLivPrb_t,
    where CumLivPrb_t = prod_{k=0}^{t} LivPrb_k (with the model's convention).
    """
    # Normalize DiscFac to list of length T
    T = getattr(agent, 'T_cycle', None)
    if T is None:
        # Fallback if not set; assume length of LivPrb
        liv_list = agent.LivPrb if isinstance(agent.LivPrb, list) else [float(agent.LivPrb)]
        T = len(liv_list)

    beta = agent.DiscFac
    if isinstance(beta, list):
        beta_list = [float(b) for b in beta]
        if len(beta_list) != T:
            beta_list = [float(beta_list[0])] * T
    else:
        beta_list = [float(beta)] * T

    liv = agent.LivPrb
    if isinstance(liv, list):
        liv_list = [float(x) for x in liv]
        if len(liv_list) != T:
            # If mismatch, broadcast first value
            liv_list = [float(liv_list[0])] * T
    else:
        liv_list = [float(liv)] * T

    # Cumulative products
    S = 0.0
    beta_prod = 1.0
    cum_liv = 1.0
    for t in range(T):
        cum_liv *= liv_list[t]
        if t == 0:
            S += 1.0 * cum_liv
        else:
            beta_prod *= beta_list[t-1]
            S += beta_prod * cum_liv
    return float(S)
