import numpy as np


def _crra_utility(c, rho):
    c = float(c)
    if rho == 1.0:
        return np.log(c)
    return (c ** (1.0 - rho)) / (1.0 - rho)


def compute_value_function_bellman(agent, m_grid, max_iter=100, tol=1e-6):
    """
    Compute V(m) via Bellman iteration on a fixed m_grid for a solved agent.
    Assumes infinite-horizon PY with a single-period solution at index 0.
    """
    c_func = agent.solution[0].cFunc
    beta = float(agent.DiscFac if not isinstance(agent.DiscFac, list) else agent.DiscFac[0])
    R = float(agent.Rfree if not isinstance(agent.Rfree, list) else agent.Rfree[0])
    Gamma = float(agent.PermGroFac if not isinstance(agent.PermGroFac, list) else agent.PermGroFac[0])
    rho = float(agent.CRRA)
    dist = agent.IncShkDstn[0]

    probs = np.asarray(dist.pmv, dtype=float)
    perm_shocks = np.asarray(dist.atoms[0], dtype=float)
    tran_shocks = np.asarray(dist.atoms[1], dtype=float)

    m_grid = np.asarray(m_grid, dtype=float)
    V_old = np.zeros_like(m_grid)
    V_new = np.zeros_like(m_grid)

    for _ in range(max_iter):
        # Bellman update over grid
        for i, m in enumerate(m_grid):
            c = float(c_func(float(m)))
            a = float(m - c)
            EV_next = 0.0
            # Expected continuation value using linear interpolation over current V_old
            for p, psi, theta in zip(probs, perm_shocks, tran_shocks):
                m_next = (R / (Gamma * float(psi))) * a + float(theta)
                # Linear interpolation with numpy.interp, allow linear extrapolation by clamping slope
                if m_next <= m_grid[0]:
                    Vn = V_old[0]
                elif m_next >= m_grid[-1]:
                    Vn = V_old[-1]
                else:
                    Vn = np.interp(m_next, m_grid, V_old)
                EV_next += float(p) * float(Vn)
            V_new[i] = _crra_utility(c, rho) + beta * EV_next
        # Convergence check
        if np.max(np.abs(V_new - V_old)) < tol:
            return m_grid, V_new
        V_old, V_new = V_new, V_old  # reuse arrays without realloc

    return m_grid, V_old


def create_value_function_interpolator(m_grid, V_grid):
    """
    Return a simple linear interpolator V(m) using numpy.interp.
    Extrapolates flatly beyond the grid endpoints.
    """
    m_grid = np.asarray(m_grid, dtype=float)
    V_grid = np.asarray(V_grid, dtype=float)

    def V_func(m):
        m = float(m)
        if m <= m_grid[0]:
            return float(V_grid[0])
        if m >= m_grid[-1]:
            return float(V_grid[-1])
        return float(np.interp(m, m_grid, V_grid))

    return V_func


def compute_newborn_EV_PY(agent, V_func):
    """
    Newborn EV in PY given a value function V(m). Uses a_nrm=0 → m=theta.
    """
    dist = agent.IncShkDstn[0]
    probs = np.asarray(dist.pmv, dtype=float)
    tran_shocks = np.asarray(dist.atoms[1], dtype=float)
    EV = 0.0
    for p, theta in zip(probs, tran_shocks):
        EV += float(p) * float(V_func(float(theta)))
    return float(EV)


def compute_population_welfare_custom(population, het_weights, base_type_count, m_grid):
    """
    For each return type block, compute Bellman V, newborn EV, and aggregate.
    """
    if base_type_count <= 0:
        raise ValueError("base_type_count must be positive")

    num_agents = len(population)
    if num_agents % base_type_count != 0:
        raise ValueError("Population size is not a multiple of base_type_count.")

    num_types = num_agents // base_type_count
    if het_weights is None or len(het_weights) != num_types:
        raise ValueError("het_weights must match number of return types")

    W_vec = np.zeros(num_types, dtype=float)

    for t in range(num_types):
        start = t * base_type_count
        end = start + base_type_count
        block = population[start:end]
        evs = []
        for ag in block:
            m_vals, V_vals = compute_value_function_bellman(ag, m_grid)
            V_func = create_value_function_interpolator(m_vals, V_vals)
            evs.append(compute_newborn_EV_PY(ag, V_func))
        W_vec[t] = float(np.mean(evs)) if len(evs) > 0 else np.nan

    het_weights = np.asarray(het_weights, dtype=float)
    W_avg_pmv = float(np.dot(W_vec, het_weights))

    # AgentCount-weighted aggregate
    type_counts = np.zeros(num_types, dtype=float)
    for t in range(num_types):
        start = t * base_type_count
        end = start + base_type_count
        type_counts[t] = float(np.sum([float(getattr(ag, 'AgentCount', 0.0)) for ag in population[start:end]]))
    total_counts = float(np.sum(type_counts))
    if total_counts > 0.0:
        W_avg_counts = float(np.dot(W_vec, type_counts / total_counts))
    else:
        W_avg_counts = float('nan')

    return W_vec, W_avg_pmv, W_avg_counts


def ih_discount_sum_custom(agent):
    beta = float(agent.DiscFac if not isinstance(agent.DiscFac, list) else agent.DiscFac[0])
    liv = float(agent.LivPrb if not isinstance(agent.LivPrb, list) else agent.LivPrb[0])
    eff = beta * liv
    if eff >= 1.0:
        raise ValueError("beta*LivPrb >= 1; discount sum diverges")
    return 1.0 / (1.0 - eff)


def consumption_equivalent_delta_custom(W_A, W_B, rho, S=None):
    rho = float(rho)
    W_A = float(W_A)
    W_B = float(W_B)
    if abs(rho - 1.0) < 1e-12:
        if S is None:
            raise ValueError("Must provide S for log utility")
        from math import exp
        return exp((W_B - W_A) / float(S)) - 1.0
    return (W_B / W_A) ** (1.0 / (1.0 - rho)) - 1.0


def diagnose_agent_value_computation(agent, m_grid):
    """
    Produce diagnostics for value-function computation for a single agent.
    Returns a dict with key stats and the newborn EV implied by the computed V.
    """
    # Core parameters
    rho = float(agent.CRRA)
    beta = float(agent.DiscFac if not isinstance(agent.DiscFac, list) else agent.DiscFac[0])
    R = float(agent.Rfree if not isinstance(agent.Rfree, list) else agent.Rfree[0])
    Gamma = float(agent.PermGroFac if not isinstance(agent.PermGroFac, list) else agent.PermGroFac[0])
    liv = float(agent.LivPrb if not isinstance(agent.LivPrb, list) else agent.LivPrb[0])

    # Shocks
    dist = agent.IncShkDstn[0]
    probs = np.asarray(dist.pmv, dtype=float)
    perm_shocks = np.asarray(dist.atoms[0], dtype=float)
    tran_shocks = np.asarray(dist.atoms[1], dtype=float)
    psi_mean = float(np.dot(perm_shocks, probs))
    theta_mean = float(np.dot(tran_shocks, probs))

    # c(m) samples
    c_func = agent.solution[0].cFunc
    m_grid = np.asarray(m_grid, dtype=float)
    m_samp_idx = np.linspace(0, len(m_grid) - 1, num=min(5, len(m_grid)), dtype=int)
    m_samples = m_grid[m_samp_idx]
    c_samples = [float(c_func(float(m))) for m in m_samples]

    # Compute V on grid
    m_vals, V_vals = compute_value_function_bellman(agent, m_grid)
    V_min = float(np.min(V_vals))
    V_max = float(np.max(V_vals))
    V_mean = float(np.mean(V_vals))

    # Newborn EV
    V_func = create_value_function_interpolator(m_vals, V_vals)
    EV_newborn = compute_newborn_EV_PY(agent, V_func)

    diag = {
        'rho': rho,
        'beta': beta,
        'R': R,
        'Gamma': Gamma,
        'LivPrb': liv,
        'psi_mean': psi_mean,
        'theta_mean': theta_mean,
        'm_min': float(m_grid[0]),
        'm_max': float(m_grid[-1]),
        'c_samples': [(float(m), float(c)) for m, c in zip(m_samples, c_samples)],
        'V_min': V_min,
        'V_max': V_max,
        'V_mean': V_mean,
        'EV_newborn': float(EV_newborn)
    }
    return diag


# ========================= Life-Cycle (finite horizon) helpers =========================

def _as_list_len_T(val, T):
    if isinstance(val, list):
        if len(val) == T:
            return [float(x) for x in val]
        elif len(val) == 1:
            return [float(val[0])] * T
        else:
            return [float(val[0])] * T
    else:
        return [float(val)] * T


def compute_value_functions_lc(agent, m_grid):
    """
    Compute lifecycle value functions {V_t(m)} backward using agent.solution[t].cFunc.
    Returns a list of callables V_funcs[t](m). Assumes terminal value at T is 0.
    """
    T = int(agent.T_cycle)
    m_grid = np.asarray(m_grid, dtype=float)
    rho = float(agent.CRRA)
    beta_list = _as_list_len_T(getattr(agent, 'DiscFac', 0.96), T)
    R_list = _as_list_len_T(getattr(agent, 'Rfree', 1.02), T)
    G_list = _as_list_len_T(getattr(agent, 'PermGroFac', 1.0), T)

    # Prepare storage for V grids
    V_grids = [np.zeros_like(m_grid) for _ in range(T)]

    # Continuation value at t=T is zero
    V_next = np.zeros_like(m_grid)

    # Backward induction
    for t in reversed(range(T)):
        c_func_t = agent.solution[t].cFunc
        beta_t = float(beta_list[t])
        R_t = float(R_list[t])
        G_t = float(G_list[t])
        dist_t = agent.IncShkDstn[t]
        probs = np.asarray(dist_t.pmv, dtype=float)
        psi = np.asarray(dist_t.atoms[0], dtype=float)
        theta = np.asarray(dist_t.atoms[1], dtype=float)

        V_t = np.zeros_like(m_grid)
        for i, m in enumerate(m_grid):
            c = float(c_func_t(float(m)))
            a = float(m - c)
            EV_next = 0.0
            for p, psi_i, th_i in zip(probs, psi, theta):
                m_next = (R_t / (G_t * float(psi_i))) * a + float(th_i)
                if m_next <= m_grid[0]:
                    Vn = V_next[0]
                elif m_next >= m_grid[-1]:
                    Vn = V_next[-1]
                else:
                    Vn = np.interp(m_next, m_grid, V_next)
                EV_next += float(p) * float(Vn)
            V_t[i] = _crra_utility(c, rho) + beta_t * EV_next
        V_grids[t] = V_t
        V_next = V_t

    # Build interpolators
    V_funcs = [create_value_function_interpolator(m_grid, V_grids[t]) for t in range(T)]
    return V_funcs, V_grids


def compute_newborn_EV_LC(agent, V_funcs):
    """
    Newborn EV in LC at entry t=0 using V_0(m0).
    Integrates over initial assets (from history/aNrmInit) and first-period shocks.
    """
    # Initial asset distribution (normalized)
    a_vals = None
    a_wts = None
    try:
        aLvl0 = agent.history['aLvl'][0, :]
        pLvl0 = agent.history['pLvl'][0, :]
        a_vals_hist = (np.asarray(aLvl0, dtype=float) / np.asarray(pLvl0, dtype=float)).flatten()
        if 'WeightFac' in agent.history:
            w0 = np.asarray(agent.history['WeightFac'][0, :], dtype=float).flatten()
            a_wts_hist = w0 / np.sum(w0) if np.sum(w0) > 0 else np.ones_like(a_vals_hist) / a_vals_hist.size
        else:
            a_wts_hist = np.ones_like(a_vals_hist) / a_vals_hist.size
        a_vals = a_vals_hist
        a_wts = a_wts_hist
    except Exception:
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

    # Period-0 parameters and shocks
    R0 = float(agent.Rfree[0] if isinstance(agent.Rfree, list) else agent.Rfree)
    G0 = float(agent.PermGroFac[0] if isinstance(agent.PermGroFac, list) else agent.PermGroFac)
    dist0 = agent.IncShkDstn[0]
    probs0 = np.asarray(dist0.pmv, dtype=float)
    psi0 = np.asarray(dist0.atoms[0], dtype=float)
    th0 = np.asarray(dist0.atoms[1], dtype=float)

    V0 = V_funcs[0]
    EV = 0.0
    for a_weight, a0 in zip(a_wts, a_vals):
        for p, psi, theta in zip(probs0, psi0, th0):
            m0 = (R0 / (G0 * float(psi))) * float(a0) + float(theta)
            EV += float(a_weight) * float(p) * float(V0(float(m0)))
    return float(EV)


def lc_discount_sum_custom(agent):
    """
    Life-cycle discount sum S_e for log utility across T periods.
    """
    T = int(getattr(agent, 'T_cycle', 1))
    beta_list = _as_list_len_T(getattr(agent, 'DiscFac', 0.96), T)
    liv_list = _as_list_len_T(getattr(agent, 'LivPrb', 1.0), T)
    S = 0.0
    beta_prod = 1.0
    cum_liv = 1.0
    for t in range(T):
        cum_liv *= float(liv_list[t])
        if t == 0:
            S += 1.0 * cum_liv
        else:
            beta_prod *= float(beta_list[t-1])
            S += beta_prod * cum_liv
    return float(S)


