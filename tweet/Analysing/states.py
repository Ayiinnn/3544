import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def create_state_transition_matrix(df):

    df['state'] = df.iloc[:, 1:5].apply(tuple, axis=1)


    transitions = pd.DataFrame({
        'current': df['state'].iloc[:-1],
        'next': df['state'].shift(-1).iloc[:-1]
    })

    all_states = sorted(df['state'].unique())
    print(all_states)

    state_index = {state: i for i, state in enumerate(all_states)}
    matrix = np.zeros((len(all_states), len(all_states)))

    for _, row in transitions.iterrows():
        current = state_index[row['current']]
        next_ = state_index[row['next']]
        matrix[current][next_] += 1

    row_sums = matrix.sum(axis=1)
    transition_matrix = matrix / row_sums[:, np.newaxis]
    transition_matrix = np.nan_to_num(transition_matrix)

    return pd.DataFrame(
        transition_matrix,
        index=all_states,
        columns=all_states
    )





def visualize_transition_matrix(transition_df, figsize=(8, 5)):

    plt.rcParams['axes.unicode_minus'] = False


    abbrev = {
        'stprice': {1: 'st↑', -1: 'st↓', 0: 'st→'},
        'ltprice': {1: 'lt↑', 0: 'lt→', -1: 'lt↓'},
        'marketsentiment': {0: 'mar→', 1: 'mar↑', -1: 'mar↓'},
        'mediasentiment': {1: 'med↑', -1: 'med↓', 0: 'med→'}
    }


    def compact_label(state):
        return '/'.join([
            abbrev['stprice'][state[0]],
            abbrev['ltprice'][state[1]],
            abbrev['marketsentiment'][state[2]],
            abbrev['mediasentiment'][state[3]]
        ])

    labels = [compact_label(s) for s in transition_df.index]
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        transition_df.values,
        annot=False,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    ax.set_xticklabels(
        labels,
        rotation=90,
        fontsize=4,
        ha='center'
    )
    ax.set_yticklabels(
        labels,
        rotation=0,
        fontsize=4
    )
    plt.title("transition matrix", fontsize=10)
    plt.xlabel("future", fontsize=10)
    plt.ylabel("current", fontsize=10)
    plt.show()
    return ax

df=pd.read_csv('final2.csv')
df2=create_state_transition_matrix(df)
ax = visualize_transition_matrix(df2)
df2.to_csv('result.csv')
print(df2)
