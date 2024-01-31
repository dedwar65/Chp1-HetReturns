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

SpecificationFilename = '.yaml'

# Choose basic specification parameters; this content will be in a YAML file later
MyAgentType = IndShockConsumerType
HetParam = 'Rfree'
DstnType = Uniform
InitCenter = 1.0105
InitSpread = 0.02
TypeCount = 7
TotalAgentCount = 70000
ModelType = 'dist'

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
MyPopulation = AgentPopulation(MyAgentType, BaseParamDict)
MyPopulation.approx_distributions({HetParam : TypeCount})
MyPopulation.create_distributed_agents()
MyPopulation.AgentCount = TotalAgentCount

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
