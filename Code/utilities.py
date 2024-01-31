import numpy as np

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
