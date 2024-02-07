'''
This file specifies parameters for the agent types and empirical targets.
'''
import numpy as np
import csv
import matplotlib.pyplot as plt
from copy import deepcopy
from HARK.core import AgentPopulation
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.distribution import Uniform, Lognormal
import pandas as pd
from HARK.Calibration.Income.IncomeTools import (
    CGM_income,
    parse_income_spec,
    parse_time_params,
)
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools import income_wealth_dists_from_scf

SpecificationFilename = '.yaml'

# Choose basic specification parameters; this content will be in a YAML file later
MyAgentType = IndShockConsumerType
HetParam = 'Rfree'
DstnType = Uniform
InitCenter = 1.0277277887072713
InitSpread = 0.02
TypeCount = 7
TotalAgentCount = 70000
ModelType = 'dist'
LifeCycle = False

# Define a baseline parameter dictionary; this content will be in a YAML file later
BaseParamDict = {
    "CRRA": 1.0,  # Coefficient of relative risk aversion
    "Rfree": 1.01 / (1.0 - 1.0 / 160.0),  # Survival probability,
    # Permanent income growth factor (no perm growth),
    "PermGroFac": [1.000**0.25],
    "PermGroFacAgg": 1.0,
    "BoroCnstArt": 0.0,
    "CubicBool": False,
    "vFuncBool": False,
    "PermShkStd": [
        (0.01 * 4 / 11) ** 0.5
    ],  # Standard deviation of permanent shocks to income
    "PermShkCount": 5,  # Number of points in permanent income shock grid
    "TranShkStd": [
        (0.01 * 4) ** 0.5
    ],  # Standard deviation of transitory shocks to income,
    "TranShkCount": 5,  # Number of points in transitory income shock grid
    "UnempPrb": 0.07,  # Probability of unemployment while working
    "IncUnemp": 0.15,  # Unemployment benefit replacement rate
    "UnempPrbRet": 0.07,
    "IncUnempRet": 0.15,
    "aXtraMin": 0.00001,  # Minimum end-of-period assets in grid
    "aXtraMax": 40,  # Maximum end-of-period assets in grid
    "aXtraCount": 32,  # Number of points in assets grid
    "aXtraExtra": [None],
    "aXtraNestFac": 3,  # Number of times to 'exponentially nest' when constructing assets grid
    "LivPrb": [1.0 - 1.0 / 160.0],  # Survival probability
    "DiscFac": 0.97,  # Default intertemporal discount factor; dummy value, will be overwritten
    "cycles": 0,
    "T_cycle": 1,
    "T_retire": 0,
    # Number of periods to simulate (idiosyncratic shocks model, perpetual youth)
    "T_sim": 300,
    "T_age": 400,
    "IndL": 10.0 / 9.0,  # Labor supply per individual (constant),
    "aNrmInitMean": np.log(0.00001),
    "aNrmInitStd": 0.0,
    "pLvlInitMean": 0.0,
    "pLvlInitStd": 0.0,
}

birth_age = 25
death_age = 90
adjust_infl_to = 2004 #same wave as the wave of SCF used for empirical targets
income_calib = CGM_income

# Define dictionaries for life cycle version of the model. Should also be in Yaml file
# Note: missing survival probabilites conditional on education level.
nohs_dict = deepcopy(BaseParamDict)
income_params = parse_income_spec(
    age_min=birth_age,
    age_max=death_age,
    adjust_infl_to=adjust_infl_to,
    **income_calib["NoHS"],
    SabelhausSong=True,
)
dist_params = income_wealth_dists_from_scf(
    base_year=adjust_infl_to, age=birth_age, education="NoHS", wave=1995
)
liv_prb = parse_ssa_life_table(
    female=True, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age - 1
)
time_params = parse_time_params(age_birth=birth_age, age_death=death_age)
nohs_dict.update(time_params)
nohs_dict.update(dist_params)
nohs_dict.update(income_params)
nohs_dict.update({"LivPrb": liv_prb})

hs_dict = deepcopy(BaseParamDict)
income_params = parse_income_spec(
    age_min=birth_age,
    age_max=death_age,
    adjust_infl_to=adjust_infl_to,
    **income_calib["HS"],
    SabelhausSong=True,
)
dist_params = income_wealth_dists_from_scf(
    base_year=adjust_infl_to, age=birth_age, education="HS", wave=1995
)
liv_prb = parse_ssa_life_table(
    female=True, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age - 1
)
time_params = parse_time_params(age_birth=birth_age, age_death=death_age)
hs_dict.update(time_params)
hs_dict.update(dist_params)
hs_dict.update(income_params)
hs_dict.update({"LivPrb": liv_prb})

college_dict = deepcopy(BaseParamDict)
income_params = parse_income_spec(
    age_min=birth_age,
    age_max=death_age,
    adjust_infl_to=adjust_infl_to,
    **income_calib["College"],
    SabelhausSong=True,
)
dist_params = income_wealth_dists_from_scf(
    base_year=adjust_infl_to, age=birth_age, education="College", wave=1995
)
liv_prb = parse_ssa_life_table(
    female=True, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age - 1
)
time_params = parse_time_params(age_birth=birth_age, age_death=death_age)
college_dict.update(time_params)
college_dict.update(dist_params)
college_dict.update(income_params)
college_dict.update({"LivPrb": liv_prb})

# Define a mapping from (center,spread) to the actual parameters of the distribution.
# For each class of distributions you want to allow, there needs to be an entry for
# DstnParam mapping that says what (center,spread) represents for that distribution.
if DstnType is Uniform:
    DstnParamMapping = lambda center, spread : [center-spread, center+spread]
elif DstnType is Lognormal:
    DstnParamMapping = lambda center, spread : [np.log(center) - 0.5 * spread**2, spread]
else:
    print('Oh no! You picked an invalid distribution type!')

# Setup basics for computing empirical targets from the SCF
TargetPercentiles = [0.2, 0.4, 0.6, 0.8]
data_location = '/Users/dc/Library/CloudStorage/OneDrive-JohnsHopkins/research/GitHub/dedwar65/Chp1-HetReturns/Code/Data'
wealth_data_file = 'rscfp2004_reduced.txt'
wealth_col = 4
weight_col = 0
income_col = 1

# Main executions of the file happen from this point onward 
# Make a population of agents with baseline parameters
BaseParamDict[HetParam] = DstnType(*DstnParamMapping(InitCenter, InitSpread))
BasePopulation = AgentPopulation(MyAgentType, BaseParamDict)
BasePopulation.approx_distributions({HetParam : TypeCount})
BasePopulation.create_distributed_agents()
BasePopulation.AgentCount = TotalAgentCount/3

# Make a population of life cycle agents with het returns
population = []
for i, subpop in enumerate([nohs_dict, hs_dict, college_dict]):
    subpop[HetParam] = DstnType(*DstnParamMapping(InitCenter, InitSpread))
    population.append(AgentPopulation(MyAgentType, subpop))
    population[i].approx_distributions({HetParam : TypeCount})
    population[i].create_distributed_agents()
    population[i].AgentCount = TotalAgentCount/3

agents = population[0].agents + population[1].agents + population[2].agents
LifeCyclePopulation = AgentPopulation(MyAgentType, {})
LifeCyclePopulation.agents = agents
LifeCyclePopulation.AgentCount = TotalAgentCount

# Import the wealth and income data to be matched in estimation
f = open(data_location + "/" + wealth_data_file)
wealth_data_reader = csv.reader(f, delimiter="\t")
wealth_data_raw = list(wealth_data_reader)
wealth_data = np.zeros(len(wealth_data_raw)) + np.nan
weights_data = deepcopy(wealth_data)
income_data = deepcopy(wealth_data)
for j in range(len(wealth_data_raw)):
    # skip the row of headers
    wealth_data[j] = float(wealth_data_raw[j][wealth_col])
    weights_data[j] = float(wealth_data_raw[j][weight_col])
    income_data[j] = float(wealth_data_raw[j][income_col])


MyPopulation = population[0]