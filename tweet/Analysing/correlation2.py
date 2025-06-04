import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('relation.csv')
base_cols = [
    'finance1',  #price(close)
    #'finance2',  #st_vola
    #'finance3',  #lt_vola
    #'finance4',  #st_trend
    #'finance5'   #lt_trend
]
y_col = 'fg'
z_col = 'tt'
min_lead = -2000
max_lead = 500
step_size = 100

lead_fg_range = np.arange(min_lead, max_lead + 1, step_size)
lead_tt_range = np.arange(min_lead, max_lead + 1, step_size)

weighted_r2_matrix = np.full((len(lead_fg_range), len(lead_tt_range)), np.nan)

#PCA
scaler = StandardScaler()
pca = PCA()
scaled_data = scaler.fit_transform(df[base_cols].dropna())
pca.fit(scaled_data)

component_weights = pca.explained_variance_ratio_
column_weights = np.abs(pca.components_).T @ component_weights
column_weights /= column_weights.sum()

#column_weights[0]+=0.1
print("weights: ")
#column_weights[0]+=0.1
#column_weights[1]+=0.2
for col, w in zip(base_cols, column_weights):
    print(f"{col}: {w:.3f}")

for i, lead_fg in enumerate(lead_fg_range):
    for j, lead_tt in enumerate(lead_tt_range):

        fg_shifted = df[y_col].shift(-lead_fg)
        tt_shifted = df[z_col].shift(-lead_tt)

        valid_data = pd.DataFrame({
            **{col: df[col] for col in base_cols},
            'fg_shifted': fg_shifted,
            'tt_shifted': tt_shifted
        }).dropna()

        if len(valid_data) < 5:
            continue

        X = valid_data[['fg_shifted', 'tt_shifted']]
        y = scaler.transform(valid_data[base_cols])

        #regression
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        individual_r2 = [
            1 - np.var(y[:, k] - y_pred[:, k]) / np.var(y[:, k])
            for k in range(y.shape[1])
        ]
        #R²
        weighted_r2 = np.dot(individual_r2, column_weights)
        weighted_r2_matrix[i, j] = weighted_r2

max_idx = np.nanargmax(weighted_r2_matrix)
optimal_fg_lead, optimal_tt_lead = np.unravel_index(max_idx, weighted_r2_matrix.shape)

print(f"\nbest: {y_col}leading{-lead_fg_range[optimal_fg_lead]}units，"
      f"{z_col}leading{-lead_tt_range[optimal_tt_lead]}units")
print(f"weighted average R²: {weighted_r2_matrix[optimal_fg_lead, optimal_tt_lead]:.3f}")

#visuallize
plt.figure(figsize=(12, 8))
sns.heatmap(
    weighted_r2_matrix,
    annot=True,
    fmt=".2f",
    xticklabels=[f"{-x}" for x in lead_tt_range],
    yticklabels=[f"{-y}" for y in lead_fg_range],
    cmap='coolwarm',
    mask=np.isnan(weighted_r2_matrix)
)
plt.title(f'leading analysis step={step_size}')
plt.xlabel(f'{z_col}lead')
plt.ylabel(f'{y_col}lead')
plt.show()