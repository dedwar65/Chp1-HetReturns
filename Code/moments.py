import pandas as pd
import numpy as np
from pathlib import Path
from utilities import (get_lorenz_shares)

# 1. Load your combined SCF CSV
csv_path = Path('Data/scf_processed_dc.csv')
df = pd.read_csv(csv_path)

# Filter the data to exclude ages greater than 70
df = df[df['age'] <= 70]

# 2. Define the percentiles and age bins
percentiles = [0.2, 0.4, 0.6, 0.8]

# 5-year age bins
age_bins_5 = np.arange(20, 71, 5)  # Change the upper limit to 71 to include 70
age_labels_5 = [f"{i}-{i+5}" for i in age_bins_5[:-1]]
df['age_bin_5yr'] = pd.cut(df['age'], bins=age_bins_5, labels=age_labels_5, right=False)

# 10-year age bins
age_bins_10 = np.arange(20, 71, 10)  # Change the upper limit to 71 to include 70
age_labels_10 = [f"{i}-{i+10}" for i in age_bins_10[:-1]]
df['age_bin_10yr'] = pd.cut(df['age'], bins=age_bins_10, labels=age_labels_10, right=False)


#Check that the age bin assignment works properly
print("\nSample of age and assigned bins:")
print(df[['age', 'age_bin_5yr', 'age_bin_10yr']].head(10))

# 3. Define a function to apply get_lorenz_shares by group
def compute_lorenz_by_group(df, value_col, weight_col, group_cols, percentiles):
    rows = []
    for keys, grp in df.groupby(group_cols):
        vals = grp[value_col].to_numpy()
        wts = grp[weight_col].to_numpy()
        shares = get_lorenz_shares(vals, wts, percentiles, presorted=False)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for p, s in zip(percentiles, shares):
            row[f'lorenz_{int(p*100)}'] = s
        rows.append(row)
    return pd.DataFrame(rows)

# Everything after this point is checking that it works properly and can be changed.
# 4. Compute Lorenz points for each year × age-bin
lorenz_5yr = compute_lorenz_by_group(df, 'networth', 'wgt', ['year', 'age_bin_5yr'], percentiles)
lorenz_10yr = compute_lorenz_by_group(df, 'networth', 'wgt', ['year', 'age_bin_10yr'], percentiles)

# 5. Display results
print("Lorenz Shares by Year & 5-year Age Bin:")
print(lorenz_5yr)

print("\nLorenz Shares by Year & 10-year Age Bin:")
print(lorenz_10yr)

# 6. Double check results
# Filter the DataFrame for the year 2004
df_2004 = df[df['year'] == 2004]

# Now compute Lorenz shares by group for the 2004 year and 5-year age bin
lorenz_5yr_2004 = compute_lorenz_by_group(df_2004, 'networth', 'wgt', ['year', 'age_bin_5yr'], percentiles)

# Compute Lorenz shares by year (for just 2004)
lorenz_2004 = compute_lorenz_by_group(df_2004, 'networth', 'wgt', ['year'], percentiles)

# Print results
print(lorenz_5yr_2004)
print(lorenz_2004)
