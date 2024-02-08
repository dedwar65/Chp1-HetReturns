from copy import copy
import numpy as np
from HARK.utilities import plot_funcs
from HARK.parallel import multi_thread_commands
from utilities import get_lorenz_shares
from parameters import MyPopulation, DstnParamMapping, HetParam, DstnType, \
                            TypeCount, TargetPercentiles, wealth_data, weights_data, \
                            income_data, InitCenter, InitSpread, LifeCycle, \
                            center_range, spread_range, emp_KY_ratio, emp_lorenz
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
    dstn = DstnType(*DstnParamMapping(center, spread)).discretize(TypeCount)
    weights = dstn.pmv
    vals = dstn.atoms
    
    for j in range(len(MyPopulation.agents)):
        setattr(MyPopulation[j], HetParam, vals[0][j])
        setattr(MyPopulation[j], 'AgentCount', int(weights[j] * MyPopulation.AgentCount))
        

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
    '''
    updateHetParamValues(center, spread)
    multi_thread_commands(MyPopulation.agents,['solve()','initialize_sim()','simulate()'])
    IndWealthArray = np.concatenate([this_type.state_now['aLvl'] for this_type in MyPopulation])
    IndProdArray = np.concatenate([this_type.state_now['pLvl'] for this_type in MyPopulation])
    return IndWealthArray, IndProdArray

# Functions to calculate simulated moments
def calc_KY_Sim(WealthDstn, ProdDstn):
    WealthToIncRatioSim = np.mean(WealthDstn) / np.mean(ProdDstn)
    #print(WealthToIncRatioSim)
    return WealthToIncRatioSim

def calc_Lorenz_Sim(WealthDstn, weights=None):
    LorenzValuesSim = get_lorenz_shares(WealthDstn, weights=weights, percentiles=TargetPercentiles)
    return LorenzValuesSim

# Intermediate functions needed for the estimation
def calc_KY_diff(center, spread):
    WealthDstn, ProdDstn = getDistributionsFromHetParamValues(center, spread)
    sim_KY_ratio = calc_KY_Sim(WealthDstn, ProdDstn)
    diff = emp_KY_ratio - sim_KY_ratio
    print(center,diff)
    return diff

def calc_Lorenz_dist(center, spread):
    WealthDstn, ProdDstn = getDistributionsFromHetParamValues(center, spread)
    sim_lorenz = calc_Lorenz_Sim(WealthDstn)
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
    MyPopulation.opt_center = opt_center
    
    dist = calc_Lorenz_dist(opt_center, spread)
    print(f"Lorenz distance found = {dist}")
    return dist

# Functions to be optimized as a part of the structural estimation
def find_Target_KY(spread=0.):
    """
    Finds the center value such that, with no heterogeneity (spread=0), the simulated
    KY ratio is equal to its empirical counterpart.
    """
    MyPopulation.opt_spread = spread
    result = root_scalar(calc_KY_diff, args=spread, method="brenth", bracket=center_range,
                xtol=10 ** (-6))
    
    opt_center = result.root
    MyPopulation.opt_center = opt_center
    return result

def min_Lorenz_dist_at_Target_KY():
    '''
    Finds the spread value such that the lorenz distance is minimized, given the 
    target KY ratio is acheived.
    '''
    result = minimize_scalar(calc_Lorenz_dist_at_Target_KY, bracket=spread_range,
                                 tol=1e-4)
    opt_spread = result.x
    MyPopulation.opt_spread = opt_spread
    return result


if __name__ == '__main__':
    from time import time
    
    t0 = time()
    # WealthDstn, ProdDstn = getDistributionsFromHetParamValues(InitCenter, InitSpread)
    # sim_target = calc_KY_Sim(WealthDstn, ProdDstn)
    # print(sim_target)
    # emp_target = calcEmpMoments(wealth_data, income_data, weights_data)
    # diff = sim_target - emp_target[0]
    # print(emp_target)
    # print(diff)
    # opt_center = root_scalar(calc_KY_diff, args=InitSpread, method="brenth", bracket=center_range,
    #             xtol=10 ** (-6)).root
    # print(opt_center)
    y = min_Lorenz_dist_at_Target_KY()
    print(y)
    t1 = time()
    
    print('That took ' + str(t1-t0) + ' seconds.')