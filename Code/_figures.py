import matplotlib.pyplot as plt
import numpy as np
from estimation import getDistributionsFromHetParamValues
from IPython.core.getipython import get_ipython
from parameters import wealth_data, weights_data
from utilities import get_lorenz_shares


def presentation_graphs(center_point, spread_point, center_dist, spread_dist):
    """
    Produces the key graph for assessing the results of the structural estimation.
    """
    # Construct the Lorenz curves from the data
    pctiles = np.linspace(0.001, 0.999, 15)  # may need to change percentiles
    SCF_lorenz = get_lorenz_shares(wealth_data, weights_data, percentiles=pctiles)

    # Construct the Lorenz curves from the simulated model
    WealthDstn_point, ProdDstn_point, WeightDstn_point, MPCDstn_point = getDistributionsFromHetParamValues(center_point, spread_point)
    Sim_lorenz_point = get_lorenz_shares(WealthDstn_point, WeightDstn_point, percentiles=pctiles)
    print(Sim_lorenz_point)

    # Construct the Lorenz curves from the simulated model
    WealthDstn_dist, ProdDstn_dist, WeightDstn_dist, MPCDstn_dist = getDistributionsFromHetParamValues(center_dist, spread_dist)
    Sim_lorenz_dist = get_lorenz_shares(WealthDstn_dist, WeightDstn_dist, percentiles=pctiles)
    print(Sim_lorenz_dist)

    # Plot
    plt.figure(figsize=(5, 5))
    plt.title("Wealth Distribution")
    plt.plot(pctiles, SCF_lorenz, "-k", label="SCF")
    plt.plot(
        pctiles, Sim_lorenz_point, "-.k", label="R-point"
    )
    plt.plot(
        pctiles, Sim_lorenz_dist, ":k", label="R-dist"
    )
    plt.plot(pctiles, pctiles, "--k", label="45 Degree")
    plt.xlabel("Percentile of net worth")
    plt.ylabel("Cumulative share of wealth")
    plt.legend(loc=2)
    plt.ylim([0, 1])
    plt.show()


presentation_graphs(1.0144198126134936, 0.0, 1.0071558412556672, 0.013311733598552632)

