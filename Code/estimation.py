from copy import copy
import numpy as np
from HARK.utilities import plot_funcs
from HARK.parallel import multi_thread_commands
from utilities import get_lorenz_shares
from parameters import MyPopulation, DstnParamMapping, HetParam, DstnType, \
                            HetTypeCount, TargetPercentiles, wealth_data, weights_data, \
                            income_data, BaseTypeCount, LifeCycle, \
                            center_range, spread_range, emp_KY_ratio, emp_lorenz
import parameters as params
from scipy.optimize import minimize, minimize_scalar, root_scalar

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
    print(sim_lorenz)
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
    
    return result

def min_Lorenz_dist_at_Target_KY():
    '''
    Finds the spread value such that the lorenz distance is minimized, given the 
    target KY ratio is acheived.
    '''
    result = minimize_scalar(calc_Lorenz_dist_at_Target_KY, bracket=spread_range,
                                 tol=1e-4)
    params.opt_spread = result.x
    return result


if __name__ == '__main__':
    from time import time
    
    t0 = time()
    y = min_Lorenz_dist_at_Target_KY()
    print(y)
    t1 = time()
    
    opt_center = params.opt_center
    opt_spread = params.opt_spread
    
    print('That took ' + str(t1-t0) + ' seconds.')