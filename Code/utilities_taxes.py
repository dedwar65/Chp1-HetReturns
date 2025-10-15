import numpy as np
import matplotlib.pyplot as plt
from utilities import get_lorenz_shares

def getDistributionsFromHetParamValues(center, spread):
    """
    Import the working function from estimation.py instead of recreating it.
    """
    from estimation import getDistributionsFromHetParamValues as estimation_getDistributions
    return estimation_getDistributions(center, spread)

def extract_income_distribution(center, spread):
    """
    Extract income and wealth distributions from the population.
    """
    WealthDstn, ProdDstn, WeightDstn, MPCDstn = getDistributionsFromHetParamValues(center, spread)
    GDP = np.dot(ProdDstn, WeightDstn)
    return GDP, WealthDstn, ProdDstn, WeightDstn

def make_Rfree_with_wealth_tax(RfreeFull, WealthTaxRate, T_cycle):
    """
    Create Rfree array with wealth tax applied.
    """
    if isinstance(RfreeFull, list):
        Rfree = RfreeFull[0]  # Get scalar value from list
    else:
        Rfree = RfreeFull

    # Apply wealth tax: Rfree_new = Rfree - WealthTaxRate
    Rfree_new = Rfree - WealthTaxRate

    # Return as list for T_cycle compatibility
    return [Rfree_new] * T_cycle

def make_Rfree_with_capital_income_tax(RfreeFull, CapitalTaxRate, T_cycle):
    """
    Create Rfree array with capital income tax applied.
    """
    # Element-wise policy: tax only positive capital income; do not subsidize when Rfree<=1
    def _cit_transform(r):
        r = float(r)
        if r > 1.0:
            return 1.0 + (r - 1.0) * (1.0 - CapitalTaxRate)
        else:
            return r

    if isinstance(RfreeFull, list):
        # If list has a single scalar, broadcast; otherwise transform element-wise
        if len(RfreeFull) == 1:
            R_val = float(RfreeFull[0])
            R_new = _cit_transform(R_val)
            return [R_new] * T_cycle
        else:
            R_list = [ _cit_transform(x) for x in RfreeFull ]
            return R_list
    else:
        R_val = float(RfreeFull)
        R_new = _cit_transform(R_val)
        return [R_new] * T_cycle

def save_lorenz_side_by_side_from_results(results_dict, tag, percentiles):
    """
    Create side-by-side Lorenz curve plots for wealth tax and capital income tax.
    Now includes SCF data for comparison.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Get Lorenz data from results_dict
    SCF_lorenz = results_dict['SCF_lorenz']
    original_lorenz = results_dict['original']['lorenz']
    wealth_tax_lorenz = results_dict['wealth_tax']['lorenz']
    capital_income_tax_lorenz = results_dict['capital_income_tax']['lorenz']

    # Ensure arrays are numpy arrays
    percentiles = np.asarray(percentiles, dtype=float)
    SCF_lorenz = np.asarray(SCF_lorenz, dtype=float)
    original_lorenz = np.asarray(original_lorenz, dtype=float)
    wealth_tax_lorenz = np.asarray(wealth_tax_lorenz, dtype=float)
    capital_income_tax_lorenz = np.asarray(capital_income_tax_lorenz, dtype=float)

    # Augment with endpoints to span [0,1]
    x_plot = np.concatenate(([0.0], percentiles, [1.0]))
    SCF_plot = np.concatenate(([0.0], SCF_lorenz, [1.0]))
    orig_plot = np.concatenate(([0.0], original_lorenz, [1.0]))
    wt_plot = np.concatenate(([0.0], wealth_tax_lorenz, [1.0]))
    cit_plot = np.concatenate(([0.0], capital_income_tax_lorenz, [1.0]))

    # Left plot: Wealth Tax
    ax1.plot(x_plot, SCF_plot, 'k-', linewidth=2, label='SCF')
    ax1.plot(x_plot, orig_plot, 'b--', linewidth=2, label='Model (Original)')
    ax1.plot(x_plot, wt_plot, 'r:', linewidth=2, label='Model (Wealth tax)')
    ax1.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='45 Degree')
    ax1.set_xlabel('Percentile of net worth')
    ax1.set_ylabel('Cumulative share of wealth')
    ax1.set_title('Wealth Tax')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc=2)

    # Right plot: Capital Income Tax
    ax2.plot(x_plot, SCF_plot, 'k-', linewidth=2, label='SCF')
    ax2.plot(x_plot, orig_plot, 'b--', linewidth=2, label='Model (Original)')
    ax2.plot(x_plot, cit_plot, 'g:', linewidth=2, label='Model (Capital income tax)')
    ax2.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='45 Degree')
    ax2.set_xlabel('Percentile of net worth')
    ax2.set_ylabel('Cumulative share of wealth')
    ax2.set_title('Capital Income Tax')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc=2)

    # Save the figure
    import os
    os.makedirs('Figures/Figures_taxes', exist_ok=True)
    filename = f'Figures/Figures_taxes/{tag}_lorenz_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Lorenz comparison plot saved to {filename}")
    return filename
