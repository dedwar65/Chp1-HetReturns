'''
This file specifies parameters for the agent types and empirical targets.
'''
import csv
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from HARK.Calibration.Income.IncomeTools import (Cagetti_income, CGM_income,
                                                 parse_income_spec,
                                                 parse_time_params)
from HARK.core import AgentPopulation
from HARK.Calibration.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.Calibration.SCF.WealthIncomeDist.SCFDistTools import \
    income_wealth_dists_from_scf
from HARK.distributions import Lognormal, Uniform
from utilities import (AltIndShockConsumerType, calcEmpMoments,
                       get_lorenz_shares)
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

MyAgentType = AltIndShockConsumerType

script_dir = os.path.dirname(os.path.abspath(__file__))
data_location = os.path.join(script_dir, '../Data/')
specs_location = os.path.join(script_dir, '../Specifications/')
SpecificationFilename = 'LCrrDistNetWorth.yaml'

with open(specs_location + SpecificationFilename, 'r') as f:
    spec_raw = f.read()
    f.close()
yaml_params = yaml.safe_load(spec_raw)
print('Loading a specification called ' + yaml_params['description'])

tag = yaml_params['tag']
model = yaml_params["model"]

# Choose basic specification parameters
HetParam = yaml_params['HetParam']
DstnTypeName = yaml_params['DstnType']
HetTypeCount = yaml_params['HetTypeCount']
TotalAgentCount = HetTypeCount*yaml_params['AgentsPerType']
LifeCycle = yaml_params['LifeCycle']

# Specify search parameters
center_range = yaml_params['center_range']
spread_range = yaml_params['spread_range']

# Setup basics for computing empirical targets from the SCF
TargetPercentiles = yaml_params['TargetPercentiles']
wealth_data_file = yaml_params['wealth_data_file']
asset_col = yaml_params['asset_col']
wealth_col = yaml_params['wealth_col']
weight_col = yaml_params['weight_col']
income_col = yaml_params['income_col']
year = yaml_params['year']

# Read full dataset and filter by year
df = pd.read_csv(os.path.join(data_location, wealth_data_file))
df_year = df[df['year'] == year]

# Extract columns by name
asset_data = df_year[asset_col].astype(float).to_numpy()
wealth_data = df_year[wealth_col].astype(float).to_numpy()
income_data = df_year[income_col].astype(float).to_numpy()
weights_data = df_year[weight_col].astype(float).to_numpy()

# Calculate empirical moments to be used as targets
empirical_moments = calcEmpMoments(asset_data, wealth_data, income_data, weights_data, TargetPercentiles)
emp_KY_ratio = empirical_moments[0]
# emp_KY_ratio = 10.26 * 4
print(emp_KY_ratio)
emp_lorenz = empirical_moments[1]
print(emp_lorenz)

# Next, series of lines of code to compute untargeted moments.
# Keep df_year unmodified for estimation targets
df_age_binned = df_year[df_year['age'] <= 70].copy()

# Age bin specifications (not aligned with simulation - includes ages 20-25)
age_bins_5 = np.arange(20, 71, 5)  # Change the upper limit to 71 to include 70
age_labels_5 = [f"{i}-{i+5}" for i in age_bins_5[:-1]]
df_age_binned['age_bin_5yr'] = pd.cut(df_age_binned['age'], bins=age_bins_5, labels=age_labels_5, right=False)

age_bins_10 = np.arange(20, 71, 10)  # Change the upper limit to 71 to include 70
age_labels_10 = [f"{i}-{i+10}" for i in age_bins_10[:-1]]
df_age_binned['age_bin_10yr'] = pd.cut(df_age_binned['age'], bins=age_bins_10, labels=age_labels_10, right=False)

# Compute empirical Lorenz shares by age bin
def compute_lorenz_by_group(df, value_col, weight_col, group_cols, percentiles):
    rows = []
    for keys, grp in df.groupby(group_cols, observed=False):  # Add observed=False here
        vals = grp[value_col].to_numpy()
        wts = grp[weight_col].to_numpy()
        shares = get_lorenz_shares(vals, wts, percentiles, presorted=False)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for p, s in zip(percentiles, shares):
            row[f'lorenz_{int(p*100)}'] = s
        rows.append(row)
    return pd.DataFrame(rows)

emp_lorenz_5yr = compute_lorenz_by_group(
    df_age_binned, value_col=wealth_col, weight_col=weight_col,
    group_cols=['age_bin_5yr'], percentiles=TargetPercentiles
)

emp_lorenz_10yr = compute_lorenz_by_group(
    df_age_binned, value_col=wealth_col, weight_col=weight_col,
    group_cols=['age_bin_10yr'], percentiles=TargetPercentiles
)

# Optional: print for verification
print("\nEmpirical Lorenz Shares by 5-Year Age Bin (Untargeted):")
print(emp_lorenz_5yr)

print("\nEmpirical Lorenz Shares by 10-Year Age Bin (Untargeted):")
print(emp_lorenz_10yr)


# Define a mapping from (center,spread) to the actual parameters of the distribution.
# For each class of distributions you want to allow, there needs to be an entry for
# DstnParam mapping that says what (center,spread) represents for that distribution.
if DstnTypeName == 'Uniform':
    DstnType = Uniform
    DstnParamMapping = lambda center, spread : [center-spread, center+spread]
elif DstnTypeName == 'Lognormal':
    DstnType = Lognormal
    DstnParamMapping = lambda center, spread : [np.log(center) - 0.5 * spread**2, spread]
else:
    print('Oh no! You picked an invalid distribution type!')

# Define a baseline parameter dictionary; this content will be in a YAML file later
base_param_filename = yaml_params['base_param_filename']
with open(specs_location + base_param_filename + '.yaml', 'r') as f:
    init_raw = f.read()
    f.close()
BaseParamDict = {
    "BaseAgentCount" : TotalAgentCount,
    "track_vars": ['aLvl','pLvl','WeightFac']
}
BaseParamDict.update(yaml.safe_load(init_raw)) # Later, add conditions to include other agent types

# Adjust survival probabilities from SSA tables using education cohort adjustments;
# method provided by Brown, Liebman, and Pollett (2002).
mort_data_file = yaml_params['mort_data_file']

birth_age = BaseParamDict['birth_age']
death_age = BaseParamDict['death_age']

# Compute base mortality rates for the specified age range
base_liv_prb = parse_ssa_life_table(
        female=True, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age - 1
    )

# Import adjustments for education and apply them to the base mortality rates
f = open(data_location + "/" + mort_data_file)
adjustment_reader = csv.reader(f, delimiter=" ")
raw_adjustments = list(adjustment_reader)
nohs_death_probs = []
hs_death_probs = []
c_death_probs = []
for j in range(death_age - birth_age):
    this_prob = 1.0 - base_liv_prb[j]
    if j < 76:
        nohs_death_probs += [1.0 - this_prob * float(raw_adjustments[j][1])]
        hs_death_probs += [1.0 - this_prob * float(raw_adjustments[j][2])]
        c_death_probs += [1.0 - this_prob * float(raw_adjustments[j][3])]
    else:
        nohs_death_probs += [1.0 - this_prob * float(raw_adjustments[75][1])]
        hs_death_probs += [1.0 - this_prob * float(raw_adjustments[75][2])]
        c_death_probs += [1.0 - this_prob * float(raw_adjustments[75][3])]

# Here define the population of agents for the simulation
if LifeCycle:
    adjust_infl_to = 2004
    income_calib = Cagetti_income

    # Define fractions of education types
    nohs_frac = 0.11
    hs_frac = 0.54
    college_frac = 0.35

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
    time_params = parse_time_params(age_birth=birth_age, age_death=death_age)
    nohs_dict.update(time_params)
    nohs_dict.update(dist_params)
    nohs_dict.update(income_params)
    nohs_dict.update({"LivPrb": nohs_death_probs})
    nohs_dict['BaseAgentCount'] = TotalAgentCount*nohs_frac

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
    time_params = parse_time_params(age_birth=birth_age, age_death=death_age)
    hs_dict.update(time_params)
    hs_dict.update(dist_params)
    hs_dict.update(income_params)
    hs_dict.update({"LivPrb": hs_death_probs})
    hs_dict['BaseAgentCount'] = TotalAgentCount*hs_frac

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
    time_params = parse_time_params(age_birth=birth_age, age_death=death_age)
    college_dict.update(time_params)
    college_dict.update(dist_params)
    college_dict.update(income_params)
    college_dict.update({"LivPrb": c_death_probs})
    college_dict['BaseAgentCount'] = TotalAgentCount*college_frac

    # Make base agent types
    DropoutType = MyAgentType(**nohs_dict)
    HighschType = MyAgentType(**hs_dict)
    CollegeType = MyAgentType(**college_dict)
    BaseTypeCount = 3
    BasePopulation = [DropoutType, HighschType, CollegeType]

else:
    IHbaseType = MyAgentType(**BaseParamDict)
    BaseTypeCount = 1
    BasePopulation = [IHbaseType]

# Set the agent population
MyPopulation = []
for n in range(HetTypeCount):
    MyPopulation += deepcopy(BasePopulation)

# Store optimal parameters here
opt_center = None
opt_spread = None
lorenz_distance = None

