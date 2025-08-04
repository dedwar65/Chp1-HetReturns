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
eff_R = 1 + 0.095
r_list_LC_2004 = [0.9755228687702358, 0.9913478974888338, 1.0071729262074318, 1.02299795492603,
             1.038822983644628, 1.054648012363226, 1.0704730410818242]
epsilon_list_LC_2004 = []

for r in r_list_LC_2004:
    epsilon = r / (eff_R - r)
    epsilon_list_LC_2004.append(epsilon)

print(epsilon_list_LC_2004)

r_list_PY_2004 = [0.9635283311463693, 0.9827521585708383, 1.0019759859953075, 1.0211998134197764,
1.0404236408442453, 1.0596474682687145, 1.0788712956931834]

epsilon_list_PY_2004 = []

for r in r_list_PY_2004:
    epsilon = r / (eff_R - r)
    epsilon_list_PY_2004.append(epsilon)

print(epsilon_list_PY_2004)

# Results for 2007

r_list_LC_2007 = [np.float64(0.9162448137084355), np.float64(0.9440322038700298),
                  np.float64(0.9718195940316241), np.float64(0.9996069841932185),
                  np.float64(1.0273943743548128), np.float64(1.0551817645164072),
                  np.float64(1.0829691546780016)]
epsilon_list_LC_2007 = []

for r in r_list_LC_2007:
    epsilon = r / (eff_R - r)
    epsilon_list_LC_2007.append(epsilon)

print("LC 2007 elasticities:", epsilon_list_LC_2007)

r_list_PY_2007 = [np.float64(0.9601479200856198), np.float64(0.9799563923815212),
                  np.float64(0.9997648646774228), np.float64(1.0195733369733242),
                  np.float64(1.0393818092692257), np.float64(1.059190281565127),
                  np.float64(1.0789987538610286)]
epsilon_list_PY_2007 = []

for r in r_list_PY_2007:
    epsilon = r / (eff_R - r)
    epsilon_list_PY_2007.append(epsilon)

print("PY 2007 elasticities:", epsilon_list_PY_2007)

# Results for 2010

r_list_LC_2010 = [np.float64(0.8757847296941473), np.float64(0.9108685475213021),
                  np.float64(0.9459523653484568), np.float64(0.9810361831756116),
                  np.float64(1.0161200010027664), np.float64(1.051203818829921),
                  np.float64(1.086287636657076)]
epsilon_list_LC_2010 = []

for r in r_list_LC_2010:
    epsilon = r / (eff_R - r)
    epsilon_list_LC_2010.append(epsilon)

print("LC 2010 elasticities:", epsilon_list_LC_2010)

r_list_PY_2010 = [np.float64(0.9237928267754795), np.float64(0.9497798704831126),
                  np.float64(0.9757669141907458), np.float64(1.001753957898379),
                  np.float64(1.027741001606012), np.float64(1.0537280453136453),
                  np.float64(1.0797150890212783)]

epsilon_list_PY_2010 = []

for r in r_list_PY_2010:
    epsilon = r / (eff_R - r)
    epsilon_list_PY_2010.append(epsilon)

print("PY 2010 elasticities:", epsilon_list_PY_2010)


# Results for 2013

r_list_LC_2013 = [np.float64(0.8720441333496496), np.float64(0.907779306483546),
                  np.float64(0.9435144796174424), np.float64(0.9792496527513388),
                  np.float64(1.0149848258852352), np.float64(1.0507199990191316),
                  np.float64(1.086455172153028)]
epsilon_list_LC_2013 = []

for r in r_list_LC_2013:
    epsilon = r / (eff_R - r)
    epsilon_list_LC_2013.append(epsilon)

print("LC 2013 elasticities:", epsilon_list_LC_2013)

r_list_PY_2013 = [np.float64(0.9201680759583459), np.float64(0.9467731481160822),
                  np.float64(0.9733782202738184), np.float64(0.9999832924315546),
                  np.float64(1.0265883645892908), np.float64(1.053193436747027),
                  np.float64(1.0797985089047633)]

epsilon_list_PY_2013 = []

for r in r_list_PY_2013:
    epsilon = r / (eff_R - r)
    epsilon_list_PY_2013.append(epsilon)

print("PY 2013 elasticities:", epsilon_list_PY_2013)


# Results for 2016

r_list_LC_2016 = [np.float64(0.8558470442621263), np.float64(0.8944586081685405),
                  np.float64(0.9330701720749546), np.float64(0.9716817359813688),
                  np.float64(1.010293299887783), np.float64(1.0489048637941971),
                  np.float64(1.0875164277006113)]
epsilon_list_LC_2016 = []

for r in r_list_LC_2016:
    epsilon = r / (eff_R - r)
    epsilon_list_LC_2016.append(epsilon)

print("LC 2016 elasticities:", epsilon_list_LC_2016)

r_list_PY_2016 = [np.float64(0.9050416940575152), np.float64(0.9342084978784342),
                  np.float64(0.9633753016993531), np.float64(0.9925421055202721),
                  np.float64(1.021708909341191), np.float64(1.05087571316211),
                  np.float64(1.080042516983029)]

epsilon_list_PY_2016 = []

for r in r_list_PY_2016:
    epsilon = r / (eff_R - r)
    epsilon_list_PY_2016.append(epsilon)

print("PY 2016 elasticities:", epsilon_list_PY_2016)


# Results for 2019

r_list_LC_2019 = [np.float64(0.8700396845564833), np.float64(0.9060673252010242),
                  np.float64(0.9420949658455652), np.float64(0.9781226064901061),
                  np.float64(1.014150247134647), np.float64(1.0501778877791879),
                  np.float64(1.0862055284237289)]
epsilon_list_LC_2019 = []

for r in r_list_LC_2019:
    epsilon = r / (eff_R - r)
    epsilon_list_LC_2019.append(epsilon)

print("LC 2019 elasticities:", epsilon_list_LC_2019)

r_list_PY_2019 = [np.float64(0.9185096923777224), np.float64(0.9453961297134126),
                  np.float64(0.9722825670491028), np.float64(0.999169004384793),
                  np.float64(1.0260554417204832), np.float64(1.0529418790561733),
                  np.float64(1.0798283163918636)]

epsilon_list_PY_2019 = []

for r in r_list_PY_2019:
    epsilon = r / (eff_R - r)
    epsilon_list_PY_2019.append(epsilon)

print("PY 2019 elasticities:", epsilon_list_PY_2019)

