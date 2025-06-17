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
eff_R = 0.02314631637939408
r_list_LC= [0.9986005904641386 - 1, 1.0153472494911955 -1, 1.0320939085182526 - 1, 1.0488405675453094 - 1, 1.0655872265723663 - 1,
          1.0823338855994233 - 1, 1.0990805446264802 - 1]
print(r_list_LC)
epsilon_list_LC = []

for r in r_list_LC:
    epsilon = r / (eff_R - r)
    epsilon_list_LC.append(epsilon)

print(epsilon_list_LC)

