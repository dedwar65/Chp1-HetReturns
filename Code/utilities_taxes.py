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
    WealthDstn, ProdDstn, WeightDstn = getDistributionsFromHetParamValues(center, spread)
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
    if isinstance(RfreeFull, list):
        Rfree = RfreeFull[0]  # Get scalar value from list
    else:
        Rfree = RfreeFull
    
    # Apply capital income tax: Rfree_new = 1 + (Rfree - 1) * (1 - CapitalTaxRate)
    # This taxes only the capital income portion (Rfree - 1)
    Rfree_new = 1 + (Rfree - 1) * (1 - CapitalTaxRate)
    
    # Return as list for T_cycle compatibility
    return [Rfree_new] * T_cycle

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

    # Left plot: Wealth Tax
    ax1.plot(percentiles, SCF_lorenz, 'k-', linewidth=2, label='SCF Data')
    ax1.plot(percentiles, original_lorenz, 'b--', linewidth=2, label='Original Model')
    ax1.plot(percentiles, wealth_tax_lorenz, 'r:', linewidth=2, label='Wealth Tax')
    ax1.plot(percentiles, percentiles, 'k:', alpha=0.5, label='45° Line')
    ax1.set_xlabel('Cumulative Population Share')
    ax1.set_ylabel('Cumulative Wealth Share')
    ax1.set_title('Wealth Tax')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Right plot: Capital Income Tax
    ax2.plot(percentiles, SCF_lorenz, 'k-', linewidth=2, label='SCF Data')
    ax2.plot(percentiles, original_lorenz, 'b--', linewidth=2, label='Original Model')
    ax2.plot(percentiles, capital_income_tax_lorenz, 'g:', linewidth=2, label='Capital Income Tax')
    ax2.plot(percentiles, percentiles, 'k:', alpha=0.5, label='45° Line')
    ax2.set_xlabel('Cumulative Population Share')
    ax2.set_ylabel('Cumulative Wealth Share')
    ax2.set_title('Capital Income Tax')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Save the figure
    import os
    os.makedirs('Figures/Figures_taxes', exist_ok=True)
    filename = f'Figures/Figures_taxes/{tag}_lorenz_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Lorenz comparison plot saved to {filename}")
    return filename
