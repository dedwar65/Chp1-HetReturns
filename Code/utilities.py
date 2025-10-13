import math
import os

import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType


class AltIndShockConsumerType(IndShockConsumerType):
    def __init__(self, **kwds):
        # 1) Initialize exactly as HARK does
        super().__init__(**kwds)
        # 2) Force scalar Rfree/LivPrb into 1‑element lists
        if not isinstance(self.Rfree, (list, tuple)):
            self.Rfree = [self.Rfree]
        if not isinstance(self.LivPrb, (list, tuple)):
            self.LivPrb = [self.LivPrb]

        # print("Patched Agent init—Rfree:", self.Rfree, "LivPrb:", self.LivPrb)

    def __repr__(self):
        return ('AltIndShockConsumerType with Rfree=' + str(self.Rfree) + ' and DiscFac=' + str(self.DiscFac))

    def sim_one_period(self):
        """
        Overwrite the core simulation routine with a simplified special one, but
        only use it for lifecycle models.
        """
        if self.cycles == 0:  # Use core simulation method if infinite horizon
            IndShockConsumerType.sim_one_period(self)
            self.state_now["WeightFac"] = self.PopGroFac ** (-self.t_age)

            # Compute MPC for infinite horizon models
            mNrmNow = self.state_now["mNrm"]
            cFuncNow = self.solution[0].cFunc  # Infinite horizon has single solution
            cNrmNow, MPCnow = cFuncNow.eval_with_derivative(mNrmNow)
            self.state_now["MPC"] = MPCnow

            return

        # If lifecycle, first deal with moving from last period's values to this period
        for var in self.state_now:
            self.state_prev[var] = self.state_now[var]

            if isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] = np.empty(self.AgentCount)
            else:
                # Probably an aggregate variable. It may be getting set by the Market.
                pass

        # First, get the age of all agents-- which is the same across all of them!
        t = self.t_cycle[0]
        N = self.AgentCount

        # Now, generate income shocks for all of the agents
        IncShkDstn = self.IncShkDstn[t - 1]
        IncShkNow = IncShkDstn.draw(N)
        PermShkNow = IncShkNow[0, :]
        TranShkNow = IncShkNow[1, :]
        PermGroFac = self.PermGroFac[t - 1]
        r_t = float(self.Rfree[t-1])
        RfreeEff = r_t / (PermGroFac * PermShkNow)
        pLvlNow = PermGroFac * PermShkNow * self.state_prev["pLvl"]

        # Move from aNrmPrev to mNrmNow using our income shock draws
        aNrmPrev = self.state_prev["aNrm"]
        bNrmNow = RfreeEff * aNrmPrev
        mNrmNow = bNrmNow + TranShkNow

        # Find consumption and the MPC for all agents
        cFuncNow = self.solution[t].cFunc
        cNrmNow, MPCnow = cFuncNow.eval_with_derivative(mNrmNow)

        # Calculate end-of-period assets in both level and normalized
        aNrmNow = mNrmNow - cNrmNow
        aLvlNow = aNrmNow * pLvlNow

        # Compute cumulative survival probability to this age
        LivPrb = np.concatenate([[1.0], self.LivPrb])
        CumLivPrb = np.prod(LivPrb[: (t + 1)])
        CohortWeight = self.PopGroFac ** (-t)
        WeightFac = CumLivPrb * CohortWeight

        # Write these results to state_now
        self.state_now["mNrm"] = mNrmNow
        self.state_now["bNrm"] = bNrmNow
        self.state_now["aNrm"] = aNrmNow
        self.state_now["pLvl"] = pLvlNow
        self.state_now["aLvl"] = aLvlNow
        self.state_now["cNrm"] = cNrmNow
        self.state_now["TranShk"] = TranShkNow
        self.state_now["MPC"] = MPCnow
        self.state_now["WeightFac"] = WeightFac * np.ones(self.AgentCount)
        self.EmpNow = np.logical_not(TranShkNow == self.IncUnemp)
        self.state_now["t_age"] = self.t_age.astype(float)

        # Advance time for all agents
        self.t_age += 1  # Age all consumers by one period
        self.t_cycle += 1  # Age all consumers within their cycle
        # Reset to zero for those who have reached the end
        self.t_cycle[self.t_cycle == self.T_cycle] = 0

# Function to calculate empirical moments
def calcEmpMoments(asset, wealth, income, weights, pctiles):
    """
    Calculate the empirical targets using the wave of the SCF specified when
    setting up the agent type.
    """
    WealthToIncRatioEmp = np.dot(asset,weights) / np.dot(income,weights)
    LorenzValuesEmp = get_lorenz_shares(wealth, weights, percentiles=pctiles)
    return WealthToIncRatioEmp, LorenzValuesEmp

def get_lorenz_shares(data, weights=None, percentiles=None, presorted=False):
    """
    Calculates the Lorenz curve at the requested percentiles of (weighted) data.
    Median by default.

    Parameters
    ----------
    data : numpy.array
        A 1D array of float data.
    weights : numpy.array
        A weighting vector for the data.
    percentiles : [float]
        A list or numpy.array of percentiles to calculate for the data.  Each element should
        be in (0,1).
    presorted : boolean
        Indicator for whether data has already been sorted.

    Returns
    -------
    lorenz_out : numpy.array
        The requested Lorenz curve points of the data.
    """
    if percentiles is None:
        percentiles = [0.5]
    else:
        if (
            not isinstance(percentiles, (list, np.ndarray))
            or min(percentiles) <= 0
            or max(percentiles) >= 1
        ):
            raise ValueError(
                "Percentiles should be a list or numpy array of floats between 0 and 1"
            )
    if weights is None:  # Set equiprobable weights if none were given
        weights = np.ones(data.size)

    if presorted:  # Sort the data if it is not already
        data_sorted = data
        weights_sorted = weights
    else:
        order = np.argsort(data)
        data_sorted = data[order]
        weights_sorted = weights[order]

    cum_dist = np.cumsum(weights_sorted) / np.sum(
        weights_sorted
    )  # cumulative probability distribution
    temp = data_sorted * weights_sorted
    cum_data = np.cumsum(temp) / sum(temp)  # cumulative ownership shares

    # Calculate the requested Lorenz shares by interpolating the cumulative ownership
    # shares over the cumulative distribution, then evaluating at requested points
    lorenz_out = np.zeros_like(percentiles)
    for i in range(lorenz_out.size):
        p = percentiles[i]
        j = np.searchsorted(cum_dist, p)
        bot = cum_dist[j-1]
        top = cum_dist[j]
        alpha = (p - bot) / (top - bot)
        lorenz_out[i] = (1.-alpha)*cum_data[j-1] + alpha*cum_data[j]

    return lorenz_out

def plot_lorenz_from_data(assets, weights, labels, percentiles=None, title=None, output_dir=None):
    """
    Plot Lorenz curves for up to three different asset series on the same graph,
    saving as 'lorenz_only.png' in the specified output_dir or alongside this module.
    """
    import matplotlib.pyplot as plt

    if len(assets) != len(labels) or len(assets) > 3:
        raise ValueError("Provide up to 3 asset series and matching labels.")
    if percentiles is None:
        percentiles = np.linspace(0.001, 0.999, 15)
    else:
        percentiles = np.asarray(percentiles)

    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        base_dir = output_dir
    os.makedirs(base_dir, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.title(title if title else "Lorenz Curve Comparison")
    plt.xlabel('Percentile of wealth')
    plt.ylabel('Cumulative share of wealth')
    plt.ylim([0, 1])

    for asset, label in zip(assets, labels):
        lorenz_vals = get_lorenz_shares(
            np.asarray(asset), np.asarray(weights), percentiles=percentiles
        )
        plt.plot(percentiles, lorenz_vals, label=label)

    # 45-degree reference line
    plt.plot(percentiles, percentiles, '--k', label='45° Line')

    # Place legend in best location to avoid overlap
    plt.legend(loc='best')

    filename = os.path.join(base_dir, 'lorenz_only.png')
    plt.savefig(filename, dpi=300)
    plt.show()



