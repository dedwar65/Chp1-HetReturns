import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import parameters as params
from HARK.parallel import multi_thread_commands
from HARK.utilities import plot_funcs
from IPython.core.getipython import get_ipython
from parameters import (BaseTypeCount, DstnParamMapping, DstnType, DstnTypeName, HetParam,
                        HetTypeCount, LifeCycle, MyPopulation,
                        TargetPercentiles, center_range, emp_KY_ratio,
                        emp_lorenz, income_data, model, spread_range, tag,
                        wealth_data, weights_data, TargetPercentiles)
from scipy.optimize import minimize, minimize_scalar, root_scalar
from utilities import get_lorenz_shares
import pandas as pd

import HARK
print("Using HARK from:", HARK.__file__)

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
# Runtime patch: make CRRA==1 safe for vFunc construction by applying the log-utility limit
try:
    import inspect
    import HARK.ConsumptionSaving.ConsIndShockModel as _CSM
    _src = inspect.getsource(_CSM.solve_one_period_ConsIndShock)
    # Replace the two exponent insertions with CRRA==1 limits (0.0)
    _src_patched = _src
    _src_patched = _src_patched.replace(
        "MPCmaxNow ** (-CRRA / (1.0 - CRRA))",
        "(0.0 if CRRA == 1.0 else MPCmaxNow ** (-CRRA / (1.0 - CRRA)))"
    )
    _src_patched = _src_patched.replace(
        "MPCminNow ** (-CRRA / (1.0 - CRRA))",
        "(0.0 if CRRA == 1.0 else MPCminNow ** (-CRRA / (1.0 - CRRA)))"
    )
    if _src_patched != _src:
        # Define the patched function in the module namespace
        _ns = {}
        exec(_src_patched, _CSM.__dict__, _ns)
        if 'solve_one_period_ConsIndShock' in _ns:
            _CSM.solve_one_period_ConsIndShock = _ns['solve_one_period_ConsIndShock']
except Exception as _e:
    # Non-fatal: if patching fails, proceed with original; other guards handle errors
    pass

# Keep references to originals
_orig_calc_limiting   = IndShockConsumerType.calc_limiting_values
_orig_describe       = IndShockConsumerType.describe_parameters
_orig_solve          = IndShockConsumerType.solve

# 1) Patch calc_limiting_values to handle edge cases
def _patched_calc_limiting(self, *args, **kwargs):
    # For infinite horizon, DiscFac should already be scalar
    # Just ensure Rfree and LivPrb lists contain floats
    for attr in ['Rfree', 'LivPrb']:
        val = getattr(self, attr, None)
        if val is not None and isinstance(val, list) and len(val) >= 1:
            setattr(self, attr, [float(val[0])])

    return _orig_calc_limiting(self, *args, **kwargs)

# 2) Patch describe_parameters to swallow any TypeError from formatting a list
def _patched_describe(self, *args, **kwargs):
    try:
        return _orig_describe(self, *args, **kwargs)
    except TypeError:
        return ""    # skip the f"{val:.5f}" mess entirely

# 3) Install the patches
IndShockConsumerType.calc_limiting_values   = _patched_calc_limiting
IndShockConsumerType.describe_parameters    = _patched_describe

# 4) Patch solve to gracefully handle CRRA==1 with vFuncBool=True
def _patched_solve(self, *args, **kwargs):
    try:
        return _orig_solve(self, *args, **kwargs)
    except ZeroDivisionError:
        # Use log-utility limit: nudge CRRA internally only for the solve
        try:
            crra_val = float(getattr(self, 'CRRA', None))
        except Exception:
            crra_val = None
        if crra_val == 1.0 and getattr(self, 'vFuncBool', False):
            had_attr = hasattr(self, '_CRRA_orig')
            if not had_attr:
                setattr(self, '_CRRA_orig', self.CRRA)
            try:
                self.CRRA = 1.0 + 1e-8  # log-utility limit, internal only
                return _orig_solve(self, *args, **kwargs)
            finally:
                # Restore exactly 1.0 for calibration/reporting
                self.CRRA = getattr(self, '_CRRA_orig', 1.0)
                if not had_attr and hasattr(self, '_CRRA_orig'):
                    delattr(self, '_CRRA_orig')
        raise

IndShockConsumerType.solve = _patched_solve

# Simple memoization cache for inner-loop KY evaluations
_ky_cache = {}

def updateHetParamValues(center, spread):
    '''
    Function that takes in (center, spread) and applies it to the AgentPopulation,
    filling in ex ante heterogeneous parameter values with current distribution.
    Changes MyPopulation.agents.

    Parameters
    ----------
    center : float
        Measure of centrality for this distribution.
    spread : float
        Measure of spread of diffusion for this distribution.

    Returns
    -------
    None
    '''
    dstn = DstnType(*DstnParamMapping(center, spread)).discretize(HetTypeCount)
    weights = dstn.pmv
    vals = dstn.atoms

    for j in range(len(MyPopulation)):
        ThisType = MyPopulation[j]
        i = j // BaseTypeCount
        setattr(ThisType, HetParam, vals[0][i])
        setattr(ThisType, 'AgentCount', int(weights[i] * ThisType.BaseAgentCount))

    for agent in MyPopulation:
        T = agent.T_cycle
        is_infinite = (agent.cycles == 0)  # Infinite horizon if cycles == 0

        for attr in ['Rfree', 'DiscFac', 'LivPrb', 'PermGroFac']:
            val = getattr(agent, attr)

            # Normalize to appropriate format
            if isinstance(val, (int, float, np.floating)):
                val = float(val)
            elif isinstance(val, np.ndarray):
                if val.size == 1:
                    val = float(val.item())
                else:
                    val = [float(v) for v in val.tolist()]
            elif isinstance(val, list):
                if len(val) == 1:
                    val = float(val[0])
                else:
                    val = [float(v) for v in val]
            else:
                raise TypeError(f"{attr} is of unsupported type {type(val)}.")

            # For infinite horizon: DiscFac stays scalar, others become single-element lists
            if is_infinite:
                if attr == 'DiscFac':
                    # DiscFac must be scalar for infinite horizon
                    if isinstance(val, list):
                        val = float(val[0]) if len(val) == 1 else val[0]
                else:
                    # Rfree, LivPrb, PermGroFac must be single-element lists
                    if not isinstance(val, list):
                        val = [float(val)]
            else:
                # For lifecycle models, expand to full length
                if isinstance(val, (int, float, np.floating)):
                    val = [float(val)] * T
                elif isinstance(val, list) and len(val) == 1 and T > 1:
                    val = [float(val[0])] * T

            # ✅ Always assign back the cleaned-up value
            setattr(agent, attr, val)

        # Add time-varying fields for lifecycle agents ONLY
        # This is the key optimization: skip expensive update() for infinite horizon
        if not is_infinite:
            agent.time_vary = ['Rfree', 'DiscFac', 'LivPrb', 'PermGroFac',
                           'IncShkDstn', 'PermShkStd', 'TranShkStd']
            agent.update()
            agent.update_income_process()


def getDistributionsFromHetParamValues(center, spread):
    '''
    Generate 1D arrays of wealth, income, weights, and MPC representing the overall
    population distribution given the center and spread of the ex ante heterogeneous parameter.

    Parameters
    ----------
    center : float
        Measure of centrality for this distribution.
    spread : float
        Measure of spread of diffusion for this distribution.

    Returns
    -------
    IndWealthArray : np.array
        Idiosyncratic wealth holdings for the entire population.
    IndProdArray : np.array
        Idiosyncratic productivity (permanent income) for the entire population.
    IndWeightArray : np.array
        Idiosyncratic agent weights for the entire population, based on cumulative
        survival probability and population growth factor.
    IndMPCArray : np.array
        Marginal propensity to consume for the entire population.
    '''
    updateHetParamValues(center, spread)

    for i, ag in enumerate(MyPopulation):
        missing = [name for name in ["IncShkDstn","PermShkStd","TranShkStd"]
                if not hasattr(ag, name)]
        if missing:
            raise RuntimeError(f"Agent {i} still missing {missing}")

    # Use HARK's joblib-based parallelization (process-safe)
    # CRRA=1 workaround is now handled automatically in solve() method
    multi_thread_commands(MyPopulation, ['solve()', 'initialize_sim()', 'simulate()'], num_jobs=4)

    if LifeCycle:
        IndWealthArray = np.concatenate([this_type.history['aLvl'].flatten() for this_type in MyPopulation])
        IndProdArray = np.concatenate([this_type.history['pLvl'].flatten() for this_type in MyPopulation])
        IndWeightArray = np.concatenate([this_type.history['WeightFac'].flatten() for this_type in MyPopulation])
        IndMPCArray = np.concatenate([this_type.history['MPC'].flatten() for this_type in MyPopulation])
    else:
        IndWealthArray = np.concatenate([this_type.state_now['aLvl'] for this_type in MyPopulation])
        IndProdArray = np.concatenate([this_type.state_now['pLvl'] for this_type in MyPopulation])
        IndWeightArray = np.concatenate([this_type.state_now['WeightFac'] for this_type in MyPopulation])
        IndMPCArray = np.concatenate([this_type.state_now['MPC'] for this_type in MyPopulation])
    return IndWealthArray, IndProdArray, IndWeightArray, IndMPCArray

# Functions to calculate simulated moments
def calc_KY_Sim(WealthDstn, ProdDstn, WeightDstn):
    WealthToIncRatioSim = np.dot(WealthDstn, WeightDstn) / np.dot(ProdDstn, WeightDstn)
    return WealthToIncRatioSim

def calc_Lorenz_Sim(WealthDstn, WeightDstn):
    LorenzValuesSim = get_lorenz_shares(WealthDstn, weights=WeightDstn, percentiles=TargetPercentiles)
    return LorenzValuesSim

def calc_MPC_by_groups(MPC_array, Wealth_array, Income_array, Weight_array, n_groups=10):
    '''
    Calculate average MPC by wealth-to-income ratio groups and by income groups.

    Parameters
    ----------
    MPC_array : np.array
        MPC values for all agents
    Wealth_array : np.array
        Wealth levels (aLvl) for all agents
    Income_array : np.array
        Income levels (pLvl) for all agents
    Weight_array : np.array
        Statistical weights for all agents
    n_groups : int
        Number of groups (10 for deciles, 5 for quintiles)

    Returns
    -------
    MPC_overall : float
        Overall weighted average MPC
    MPC_by_WY : np.array
        Average MPC for each W/Y group (bottom to top)
    MPC_by_income : np.array
        Average MPC for each income group (bottom to top)
    '''
    # Filter out any invalid values
    valid_idx = (np.isfinite(MPC_array) & np.isfinite(Wealth_array) &
                 np.isfinite(Income_array) & (Weight_array > 0) & (Income_array > 0))
    MPC = MPC_array[valid_idx]
    Wealth = Wealth_array[valid_idx]
    Income = Income_array[valid_idx]
    Weights = Weight_array[valid_idx]

    # Normalize weights
    Weights = Weights / np.sum(Weights)

    # Overall average MPC
    MPC_overall = np.dot(MPC, Weights)

    # Wealth-to-income ratio
    WealthToIncome = Wealth / Income

    # Percentile boundaries
    percentiles = np.linspace(0, 100, n_groups + 1)

    # MPC by W/Y groups (bottom to top)
    MPC_by_WY = np.zeros(n_groups)
    cutoffs_WY = np.percentile(WealthToIncome, percentiles)

    for i in range(n_groups):
        if i == 0:
            mask = WealthToIncome <= cutoffs_WY[i+1]
        elif i == n_groups - 1:
            mask = WealthToIncome > cutoffs_WY[i]
        else:
            mask = (WealthToIncome > cutoffs_WY[i]) & (WealthToIncome <= cutoffs_WY[i+1])

        if np.sum(Weights[mask]) > 0:
            group_w = Weights[mask] / np.sum(Weights[mask])
            MPC_by_WY[i] = np.dot(MPC[mask], group_w)
        else:
            MPC_by_WY[i] = np.nan

    # MPC by income groups (bottom to top)
    MPC_by_income = np.zeros(n_groups)
    cutoffs_income = np.percentile(Income, percentiles)

    for i in range(n_groups):
        if i == 0:
            mask = Income <= cutoffs_income[i+1]
        elif i == n_groups - 1:
            mask = Income > cutoffs_income[i]
        else:
            mask = (Income > cutoffs_income[i]) & (Income <= cutoffs_income[i+1])

        if np.sum(Weights[mask]) > 0:
            group_w = Weights[mask] / np.sum(Weights[mask])
            MPC_by_income[i] = np.dot(MPC[mask], group_w)
        else:
            MPC_by_income[i] = np.nan

    return MPC_overall, MPC_by_WY, MPC_by_income


def compute_MPC_statistics(center, spread, n_groups=10):
    '''
    Compute MPC statistics at given center and spread parameters.

    Parameters
    ----------
    center : float
        Center parameter value
    spread : float
        Spread parameter value
    n_groups : int
        Number of groups (10 for deciles, 5 for quintiles)

    Returns
    -------
    mpc_stats : dict
        Dictionary containing MPC statistics and group labels
    '''
    # Get distributions
    IndWealthArray, IndProdArray, IndWeightArray, IndMPCArray = getDistributionsFromHetParamValues(center, spread)

    # Calculate MPC statistics
    MPC_overall, MPC_by_WY, MPC_by_income = calc_MPC_by_groups(
        IndMPCArray, IndWealthArray, IndProdArray, IndWeightArray, n_groups=n_groups
    )

    # Create group labels
    if n_groups == 10:
        group_labels = [f'{i*10}-{(i+1)*10}th percentile' for i in range(10)]
    elif n_groups == 5:
        group_labels = ['Bottom 20%', '20-40%', '40-60%', '60-80%', 'Top 20%']
    else:
        group_labels = [f'Group {i+1}' for i in range(n_groups)]

    mpc_stats = {
        'MPC_overall': MPC_overall,
        'MPC_by_WY': MPC_by_WY,
        'MPC_by_income': MPC_by_income,
        'n_groups': n_groups,
        'group_labels': group_labels
    }

    return mpc_stats

# Intermediate functions needed for the estimation
def calc_KY_diff(center, spread):
    # Memoize by (center, spread)
    key = (float(center), float(spread))
    if key in _ky_cache:
        return _ky_cache[key]

    WealthDstn, ProdDstn, WeightDstn, MPCDstn = getDistributionsFromHetParamValues(center, spread)
    sim_KY_ratio = calc_KY_Sim(WealthDstn, ProdDstn, WeightDstn)
    diff = emp_KY_ratio - sim_KY_ratio
    _ky_cache[key] = diff
    print(center, diff)
    return diff

def calc_Lorenz_dist(center, spread):
    WealthDstn, ProdDstn, WeightDstn, MPCDstn = getDistributionsFromHetParamValues(center, spread)
    sim_lorenz = calc_Lorenz_Sim(WealthDstn, WeightDstn)
    dist = np.sum((sim_lorenz - emp_lorenz)**2)
    return dist

def calc_Lorenz_dist_at_Target_KY(spread):
    '''
    For a given spread, find the center value which matches the KY ratio from the
    data.
    '''
    print(f"function calc_Lorenz_dist_at_Target_KY Now trying spread = {spread}...")
    opt_center = root_scalar(calc_KY_diff, args=spread, method="brentq", bracket=center_range,
                xtol=1e-3, maxiter=20).root  # Very relaxed tolerance, low iteration limit
    dist = calc_Lorenz_dist(opt_center, spread)
    params.opt_center = opt_center
    print(f"Lorenz distance found = {dist}")
    return dist

# Functions to be optimized as a part of the structural estimation
def find_center_by_matching_target_KY(spread=0.):
    """
    Finds the center value such that, with no heterogeneity (spread=0), the simulated
    KY ratio is equal to its empirical counterpart.
    """
    result = root_scalar(calc_KY_diff, args=spread, method="brentq", bracket=center_range,
                xtol=1e-3, maxiter=20)  # Very relaxed tolerance, low iteration limit
    params.lorenz_distance = calc_Lorenz_dist(result.root, spread)
    params.opt_center = result.root
    params.opt_spread = spread
    return result

def min_Lorenz_dist_at_Target_KY():
    '''
    Finds the spread value such that the lorenz distance is minimized, given the
    target KY ratio is acheived.
    '''
    result = minimize_scalar(calc_Lorenz_dist_at_Target_KY, bracket=spread_range,
                                 options={'maxiter': 15})  # Low iteration limit
    params.opt_spread = result.x
    params.lorenz_distance = result
    return result

script_dir = os.path.dirname(os.path.abspath(__file__))
figures_location = os.path.join(script_dir, '../Figures/')

def get_final_simulated_lorenz(center, spread):
    '''
    Returns the simulated Lorenz points at the given center and spread,
    for comparison to the empirical Lorenz targets (untargeted by age).
    '''
    WealthDstn, ProdDstn, WeightDstn, MPCDstn = getDistributionsFromHetParamValues(center, spread)
    return get_lorenz_shares(WealthDstn, weights=WeightDstn, percentiles=TargetPercentiles)

def get_group_shares(data, weights, percentiles):
    '''
    Calculate the share of total wealth held by each group defined by percentile boundaries.
    Unlike get_lorenz_shares which returns cumulative shares, this returns the share
    held BY each group.

    Parameters
    ----------
    data : numpy.array
        A 1D array of wealth data.
    weights : numpy.array
        A weighting vector for the data.
    percentiles : list of float
        Percentile boundaries defining groups. E.g., [0.2, 0.4, 0.6, 0.8, 0.95] creates
        groups: 0-20%, 20-40%, 40-60%, 60-80%, 80-95%, 95-100%

    Returns
    -------
    group_shares : numpy.array
        Share of total wealth held by each group (sums to 1.0)
    '''
    if weights is None:
        weights = np.ones(data.size)

    # Sort data and weights by wealth
    order = np.argsort(data)
    data_sorted = data[order]
    weights_sorted = weights[order]

    # Cumulative distribution (by population)
    cum_dist = np.cumsum(weights_sorted) / np.sum(weights_sorted)

    # Weighted wealth by observation
    weighted_wealth = data_sorted * weights_sorted
    total_wealth = np.sum(weighted_wealth)

    # Find indices corresponding to percentile boundaries
    percentile_indices = [np.searchsorted(cum_dist, p) for p in percentiles]

    # Calculate share held by each group
    group_shares = []
    prev_idx = 0

    for idx in percentile_indices:
        group_wealth = np.sum(weighted_wealth[prev_idx:idx])
        group_shares.append(group_wealth / total_wealth)
        prev_idx = idx

    # Add final group (above last percentile to 100%)
    final_group_wealth = np.sum(weighted_wealth[prev_idx:])
    group_shares.append(final_group_wealth / total_wealth)

    return np.array(group_shares)

def get_wealth_shares_by_groups(center, spread):
    '''
    Returns the share of wealth held by each group (not cumulative).
    Groups: Bottom 20%, 20-40%, 40-60%, 60-80%, 80-95%, Top 5%

    Parameters
    ----------
    center : float
        Center parameter value
    spread : float
        Spread parameter value

    Returns
    -------
    group_shares : numpy.array
        Share of total wealth held by each group
    group_labels : list of str
        Labels for each group
    '''
    # Get wealth distribution
    WealthDstn, ProdDstn, WeightDstn, MPCDstn = getDistributionsFromHetParamValues(center, spread)

    # Define percentile boundaries
    percentiles = [0.2, 0.4, 0.6, 0.8, 0.95]

    # Calculate group shares directly
    group_shares = get_group_shares(WealthDstn, WeightDstn, percentiles)

    # Create labels
    group_labels = ["Bottom 20%", "20-40%", "40-60%", "60-80%", "80-95%", "Top 5%"]

    return group_shares, group_labels

def get_empirical_wealth_shares_by_groups():
    '''
    Returns the empirical share of wealth held by each group (not cumulative).
    Uses the wealth_data and weights_data from parameters (already filtered by year).
    Groups: Bottom 20%, 20-40%, 40-60%, 60-80%, 80-95%, Top 5%

    Returns
    -------
    group_shares : numpy.array
        Share of total wealth held by each group
    group_labels : list of str
        Labels for each group
    '''
    # Define percentile boundaries
    percentiles = [0.2, 0.4, 0.6, 0.8, 0.95]

    # Calculate group shares directly from empirical data
    group_shares = get_group_shares(wealth_data, weights_data, percentiles)

    # Create labels
    group_labels = ["Bottom 20%", "20-40%", "40-60%", "60-80%", "80-95%", "Top 5%"]

    return group_shares, group_labels

def get_final_KY_ratio(center, spread):
    """
    Returns the simulated aggregate K/Y (wealth–to–income) ratio
    at the given center and spread.
    """
    WealthDstn, ProdDstn, WeightDstn, MPCDstn = getDistributionsFromHetParamValues(center, spread)
    return calc_KY_Sim(WealthDstn, ProdDstn, WeightDstn)

def get_discretized_het_params(center, spread):
    """
    Returns the list of the HetParam values (length = HetTypeCount)
    implied by the uniform distribution with (center, spread).
    """
    dstn = DstnType(*DstnParamMapping(center, spread)).discretize(HetTypeCount)
    # dstn.atoms is a 2‐D array [1 × HetTypeCount], so flatten to a 1D list:
    return list(dstn.atoms.flatten())

script_dir = os.path.dirname(os.path.abspath(__file__))
results_location = os.path.join(script_dir, '../Results/')

def show_statistics(tag, center, spread, dist, mpc_stats=None, age_lorenz_df=None):
    """
    Calculate