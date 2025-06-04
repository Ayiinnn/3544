import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('relation.csv')

base_col = 'finance2'
y_col = 'fg'
z_col = 'tt'
min_lead = -2600
max_lead = 200
step_size = 100


lead_fg_range = np.arange(min_lead, max_lead + 1, step_size)
lead_tt_range = np.arange(min_lead, max_lead + 1, step_size)

r_squared_matrix = np.full((len(lead_fg_range), len(lead_tt_range)), np.nan)

for i, lead_fg in enumerate(lead_fg_range):
    for j, lead_tt in enumerate(lead_tt_range):

        fg_shifted = df[y_col].shift(-lead_fg)
        tt_shifted = df[z_col].shift(-lead_tt)


        valid_data = pd.DataFrame({
            base_col: df[base_col],
            'fg_shifted': fg_shifted,
            'tt_shifted': tt_shifted
        }).dropna()

        X = valid_data[['fg_shifted', 'tt_shifted']]
        y = valid_data[base_col]
        model = LinearRegression().fit(X, y)

        r_squared_matrix[i, j] = model.score(X, y)


max_idx = np.nanargmax(r_squared_matrix)
optimal_fg_lead, optimal_tt_lead = np.unravel_index(max_idx, r_squared_matrix.shape)

print(f"Optimal leads: {y_col}leading{-lead_fg_range[optimal_fg_lead]}units，"
      f"{z_col}leading{-lead_tt_range[optimal_tt_lead]}units")
print(f"Maximum R²: {r_squared_matrix[optimal_fg_lead, optimal_tt_lead]:.3f}")

#可视化
plt.figure(figsize=(12, 8))
sns.heatmap(
    r_squared_matrix,
    annot=True,
    fmt=".2f",
    xticklabels=[f"{-x}" for x in lead_tt_range],
    yticklabels=[f"{-y}" for y in lead_fg_range],
    cmap='coolwarm',
    mask=np.isnan(r_squared_matrix)
)

plt.title(f'correlation step={step_size}')
plt.xlabel(f'{z_col}leading time')
plt.ylabel(f'{y_col}leading time')
plt.show()