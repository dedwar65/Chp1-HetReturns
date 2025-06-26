import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from utilities import (get_lorenz_shares)
from estimation import (getDistributionsFromHetParamValues,
                        calc_Lorenz_Sim)
from parameters import (opt_center, opt_spread)


#Life cycle epsilon values for elasticites
eff_R = 0.095
r_list_LC= [0.9755228687702358 - 1, 0.9913478974888338 - 1, 1.0071729262074318 - 1, 1.02299795492603 - 1,
             1.038822983644628 - 1, 1.054648012363226 - 1, 1.0704730410818242 - 1]
print(r_list_LC)
epsilon_list_LC = []

for r in r_list_LC:
    epsilon = r / (eff_R - r)
    epsilon_list_LC.append(epsilon)

print(epsilon_list_LC)

r_list_PY = [0.9635283311463693 -1, 0.9827521585708383 -1, 1.0019759859953075 -1, 1.0211998134197764 -1,
1.0404236408442453 -1, 1.0596474682687145 -1, 1.0788712956931834 -1]

print(r_list_PY)
epsilon_list_PY = []

for r in r_list_PY:
    epsilon = r / (eff_R - r)
    epsilon_list_PY.append(epsilon)

print(epsilon_list_PY)

