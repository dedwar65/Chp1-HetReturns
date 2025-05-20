import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import parameters as params
from HARK.parallel import multi_thread_commands
from HARK.utilities import plot_funcs
from IPython.core.getipython import get_ipython
from parameters import (BaseTypeCount, DstnParamMapping, DstnType, HetParam,
                        HetTypeCount, LifeCycle, MyPopulation,
                        TargetPercentiles, center_range, emp_KY_ratio,
                        emp_lorenz, income_data, model, spread_range, tag,
                        wealth_data, weights_data, TargetPercentiles)
from scipy.optimize import minimize, minimize_scalar, root_scalar
from utilities import get_lorenz_shares, show_statistics, results_location
import pandas as pd


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


def getDistributionsFromHetParamValues(center, spread):
    '''
    Generate a 1D array of wealth and income levels representing the overall population distribution
    of wealth given the center and spread of the ex ante heterogeneous parameter.

    center : float
        Measure of centrality for this distribution.
    spread : float
        Measure of spread of diffusion for this distribution.

    Returns
    -------
    IndWealthArray : np.array
        Idiosyncratic wealth holdings for the entire population.
    IndProdArray : np.array
        Idiosyncratic productivity for the entire population.
    IndWeightArray : np.array
        Idiosyncratic agent weights for the entire population, based on cumulative
        survival probability and population growth factor.
    '''
    updateHetParamValues(center, spread)
    multi_thread_commands(MyPopulation,['solve()','initialize_sim()','simulate()'])
    if LifeCycle:
        IndWealthArray = np.concatenate([this_type.history['aLvl'].flatten() for this_type in MyPopulation])
        IndProdArray = np.concatenate([this_type.history['pLvl'].flatten() for this_type in MyPopulation])
        IndWeightArray = np.concatenate([this_type.history['WeightFac'].flatten() for this_type in MyPopulation])
    else:
        IndWealthArray = np.concatenate([this_type.state_now['aLvl'] for this_type in MyPopulation])
        IndProdArray = np.concatenate([this_type.state_now['pLvl'] for this_type in MyPopulation])
        IndWeightArray = np.concatenate([this_type.state_now['WeightFac'] for this_type in MyPopulation])
    return IndWealthArray, IndProdArray, IndWeightArray

# Functions to calculate simulated moments
def calc_KY_Sim(WealthDstn, ProdDstn, WeightDstn):
    WealthToIncRatioSim = np.dot(WealthDstn, WeightDstn) / np.dot(ProdDstn, WeightDstn)
    return WealthToIncRatioSim

def calc_Lorenz_Sim(WealthDstn, WeightDstn):
    LorenzValuesSim = get_lorenz_shares(WealthDstn, weights=WeightDstn, percentiles=TargetPercentiles)
    return LorenzValuesSim

# Intermediate functions needed for the estimation
def calc_KY_diff(center, spread):
    WealthDstn, ProdDstn, WeightDstn = getDistributionsFromHetParamValues(center, spread)
    sim_KY_ratio = calc_KY_Sim(WealthDstn, ProdDstn, WeightDstn)
    diff = emp_KY_ratio - sim_KY_ratio
    print(center,diff)
    return diff

def calc_Lorenz_dist(center, spread):
    WealthDstn, ProdDstn, WeightDstn = getDistributionsFromHetParamValues(center, spread)
    sim_lorenz = calc_Lorenz_Sim(WealthDstn, WeightDstn)
    dist = np.sum((sim_lorenz - emp_lorenz)**2)
    return dist

def calc_Lorenz_dist_at_Target_KY(spread):
    '''
    For a given spread, find the center value which matches the KY ratio from the
    data.
    '''
    print(f"function calc_Lorenz_dist_at_Target_KY Now trying spread = {spread}...")
    opt_center = root_scalar(calc_KY_diff, args=spread, method="brenth", bracket=center_range,
                xtol=10 ** (-6)).root
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
    result = root_scalar(calc_KY_diff, args=spread, method="brenth", bracket=center_range,
                xtol=10 ** (-6))
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
                                 tol=1e-4)
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
    WealthDstn, _, WeightDstn = getDistributionsFromHetParamValues(center, spread)
    return get_lorenz_shares(WealthDstn, weights=WeightDstn, percentiles=TargetPercentiles)

def graph_lorenz(center, spread):
    """
    Produces the key graph for assessing the results of the structural estimation.
    """
    # Construct the Lorenz curves from the data
    pctiles = np.linspace(0.001, 0.999, 15)  # may need to change percentiles
    SCF_lorenz = get_lorenz_shares(wealth_data, weights_data, percentiles=pctiles)

    # Construct the Lorenz curves from the simulated model
    WealthDstn, ProdDstn, WeightDstn = getDistributionsFromHetParamValues(center, spread)
    Sim_lorenz= get_lorenz_shares(WealthDstn, WeightDstn, percentiles=pctiles)

    # Plot
    plt.figure(figsize=(5, 5))
    if model == "Point":
        plt.title("No heterogeneity")
    elif model == "Dist":
        plt.title("Return heterogeneity")
    else:
        raise ValueError("Model must be either Point or Dist")
    plt.plot(pctiles, SCF_lorenz, "-k", label="SCF")
    plt.plot(
        pctiles, Sim_lorenz, "-.k", label=f"{HetParam}-{model}"
    )
    plt.plot(pctiles, pctiles, "--k", label="45 Degree")
    plt.xlabel("Percentile of net worth")
    plt.ylabel("Cumulative share of wealth")
    plt.legend(loc=2)
    plt.ylim([0, 1])
    # Save the plot to the specified file path
    if tag is not None:
        file_path = figures_location + tag + "Plot.png"
    plt.savefig(
        file_path, format="png", dpi=300
    )  # You can adjust the format and dpi as needed

    # Display plot; if running from command line, set interactive mode on, and make figure without blocking execution
    if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
        plt.show()
    else:
        plt.ioff()
        plt.show(block=False)
        # Give OS time to make the plot (it only draws when main thread is sleeping)
        plt.pause(2)

def compute_simulated_lorenz_by_age_bins(center, spread, percentiles):
    '''
    Computes Lorenz shares from simulation grouped by:
    - 5-year bins: 25–30, 30–35, ..., 65–70
    - Mixed bins: 25–30, 30–40, 40–50, ..., 60–70

    Parameters
    ----------
    center : float
        Optimal center from estimation.
    spread : float
        Optimal spread from estimation.
    percentiles : list of float
        Percentiles to compute Lorenz shares at.

    Returns
    -------
    df_5yr : pd.DataFrame of Lorenz shares by 5-year bins
    df_10yr : pd.DataFrame of Lorenz shares by hybrid 10-year bins
    '''
    updateHetParamValues(center, spread)
    multi_thread_commands(MyPopulation, ['solve()', 'initialize_sim()', 'simulate()'])

    # Define age bins
    age_bins_5yr = np.arange(25, 75 + 1, 5)
    labels_5yr = [f"{i}-{i+5}" for i in age_bins_5yr[:-1]]

    age_bins_10yr = [25, 30, 40, 50, 60, 70]
    labels_10yr = ["25-30", "30-40", "40-50", "50-60", "60-70"]

    binning = {
        '5yr': {'bins': age_bins_5yr, 'labels': labels_5yr, 'results': {}},
        '10yr': {'bins': age_bins_10yr, 'labels': labels_10yr, 'results': {}}
    }

    if LifeCycle:
        for this_type in MyPopulation:
            aLvl_hist = this_type.history['aLvl']
            wFac_hist = this_type.history['WeightFac']
            T = aLvl_hist.shape[0]

            for t in range(T):
                age = 25 + t
                if age < 25 or age >= 70:
                    continue  # Ensure only ages 25 through 69 are used

                for key in binning:
                    bin_def = binning[key]
                    bin_label = pd.cut(
                        [age],
                        bins=bin_def['bins'],
                        labels=bin_def['labels'],
                        right=False,
                        include_lowest=True
                    )[0]
                    if bin_label not in bin_def['results']:
                        bin_def['results'][bin_label] = {'wealth': [], 'weights': []}
                    bin_def['results'][bin_label]['wealth'].append(aLvl_hist[t, :])
                    bin_def['results'][bin_label]['weights'].append(wFac_hist[t, :])
    else:
        aLvl = np.concatenate([t.state_now['aLvl'] for t in MyPopulation])
        wFac = np.concatenate([t.state_now['WeightFac'] for t in MyPopulation])
        for key in binning:
            bin_label = binning[key]['labels'][0]
            binning[key]['results'][bin_label] = {'wealth': [aLvl], 'weights': [wFac]}

    def compute_df(bin_def):
        rows = []
        for bin_label, data in bin_def['results'].items():
            vals = np.concatenate(data['wealth'])
            wts = np.concatenate(data['weights'])
            shares = get_lorenz_shares(vals, wts, percentiles)
            row = {'age_bin': bin_label}
            for p, s in zip(percentiles, shares):
                row[f'lorenz_{int(p*100)}'] = s
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df[df['age_bin'].notna()]  # Drop any NaN-labeled bins
        return df

    df_5yr = compute_df(binning['5yr'])
    df_10yr = compute_df(binning['10yr'])

    return df_5yr, df_10yr

def estimation():
    """
    Performs the estimation based on the specifications from the yaml file.
    1. Produces an accompanying results file and key graph.
    2. Prints untargeted moments.
    3. Returns what the lorenz targets were for comparison.
    """
    if model == "Point":
        find_center_by_matching_target_KY()

    elif model == "Dist":
        min_Lorenz_dist_at_Target_KY()

    opt_center = params.opt_center
    opt_spread = params.opt_spread
    lorenz_dist = params.lorenz_distance

    show_statistics(tag, opt_center, opt_spread, lorenz_dist)
    graph_lorenz(opt_center, opt_spread)

    # NEW: Compute simulated Lorenz shares by age bins
    df_sim_lorenz_5yr, df_sim_lorenz_10yr = compute_simulated_lorenz_by_age_bins(
        center=opt_center,
        spread=opt_spread,
        percentiles=TargetPercentiles
    )

    print("\nSimulated Lorenz Shares by 5-Year Age Bin:")
    print(df_sim_lorenz_5yr)

    print("\nSimulated Lorenz Shares by 10-Year Age Bin:")
    print(df_sim_lorenz_10yr)

    # Also print the final simulated Lorenz shares at optimal parameters (used in estimation)
    final_sim_lorenz = get_final_simulated_lorenz(opt_center, opt_spread)

    print("\nEmpirical vs. Simulated Lorenz Shares at Optimal Parameters:")
    for p, sim_val, emp_val in zip(TargetPercentiles, final_sim_lorenz, emp_lorenz):
        print(f"  P{int(p*100)}% — Sim: {sim_val:.6f} | Emp: {emp_val:.6f}")

    # Export simulated and empirical lorenz shares to an excel file

    results_file = os.path.join(results_location, f"Lorenz_by_age_{tag}.xlsx")
    with pd.ExcelWriter(results_file, engine='xlsxwriter') as writer:
        df_sim_lorenz_5yr.to_excel(writer, sheet_name='Sim_5yr', index=False)
        df_sim_lorenz_10yr.to_excel(writer, sheet_name='Sim_10yr', index=False)
        params.emp_lorenz_5yr.to_excel(writer, sheet_name='Emp_5yr', index=False)
        params.emp_lorenz_10yr.to_excel(writer, sheet_name='Emp_10yr', index=False)

    print(f"\nExported Lorenz shares by age to: {results_file}")


if __name__ == '__main__':
    from time import time

    t0 = time()
    estimation()
    t1 = time()

    print('That took ' + str(t1-t0) + ' seconds.')
