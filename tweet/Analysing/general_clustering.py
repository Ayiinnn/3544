import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Timestamp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def loading(file_path):
    df0 = pd.read_csv(file_path, parse_dates=['timestamp'])
    df0 = df0.sort_values('timestamp').reset_index(drop=True)
    return df0
def loading_fg(file_path):
    df0 = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df0 = df0.sort_values('Timestamp').reset_index(drop=True)
    return df0

def calculate_avg_change(df):

    # st range
    df['short_term_avg'] = df['close'].rolling(window=5, center=True).mean()
    df['future_avg'] = df['close'].shift(-10).rolling(window=5, center=True).mean()
    df['short_term_change'] = (df['future_avg'] - df['short_term_avg']) / df['short_term_avg']

    # lt range
    df['long_term_avg'] = df['close'].rolling(window=9, center=True).mean()
    df['future_long_avg'] = df['close'].shift(-60).rolling(window=5, center=True).mean()
    df['long_term_change'] = (df['future_long_avg'] - df['long_term_avg']) / df['long_term_avg']

    df.drop(columns=['short_term_avg','future_avg','long_term_avg','future_long_avg'],inplace=True)
    df.fillna(0,inplace=True)
    return df

def calculate_avg_change_fg(df):

    #range
    df['long_term_avg'] = df['Value'].rolling(window=72, center=True).mean()
    df['future_long_avg'] = df['Value'].shift(-144).rolling(window=72, center=True).mean()
    df['change'] = (df['future_long_avg'] - df['long_term_avg']) / df['long_term_avg']

    df.drop(columns=['long_term_avg','future_long_avg'],inplace=True)
    df.fillna(0,inplace=True)
    return df

def calculate_avg_change_tt(df):


    df['long_term_avg'] = df['avg_senti'].rolling(window=7, center=True).mean()
    df['future_long_avg'] = df['avg_senti'].shift(-13).rolling(window=7, center=True).mean()
    df['avg_change'] = (df['future_long_avg'] - df['long_term_avg']) / df['long_term_avg']

    df.drop(columns=['long_term_avg','future_long_avg'],inplace=True)
    df.fillna(0,inplace=True)
    return df

def clusters_1(df, vol_col, col_name,time_col='timestamp'):

    X = df[[vol_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    distortions = []
    K_range = range(1, 6)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    #elbow
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')

    #k
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    plt.subplot(1, 2, 2)
    for cluster_id in range(optimal_k):
        cluster_data = X[clusters == cluster_id][vol_col]
        sns.kdeplot(cluster_data, label=f'Cluster {cluster_id}', fill=True)
    plt.xlabel(vol_col)
    plt.ylabel('Density')
    plt.title(f'Trend Distribution by Cluster (K={optimal_k})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    df_cluster = df.copy()
    df_cluster[col_name] = clusters

    plt.figure(figsize=(12, 6))
    plt.scatter(df_cluster[time_col], df_cluster[vol_col],
                c=df_cluster[col_name], cmap='viridis', alpha=0.6)
    plt.xlabel('Time')
    plt.ylabel(vol_col)
    plt.title('Cluster Distribution Over Time')
    plt.colorbar(label='Cluster')
    plt.xticks(rotation=45)
    plt.show()

    df_cluster['cluster_change'] = (df_cluster[col_name] != df_cluster[col_name].shift()).cumsum()
    state_durations = df_cluster.groupby('cluster_change').size()
    print("cluster duration:")
    print(f"- average: {state_durations.mean():.1f} ")
    print(f"- max: {state_durations.max()} ")
    print(f"- min: {state_durations.min()} ")

    df_cluster.drop(columns=['cluster_change'],inplace=True)
    return df_cluster

def clusters_2(df, vola_col1, vola_col2, time_col='timestamp'):

    X = df[[vola_col1, vola_col2]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    distortions = []
    K_range = range(1, 6)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    #elbow
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')

    #k
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)


    plt.subplot(1, 2, 2)
    plt.scatter(X[vola_col1], X[vola_col2], c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel(vola_col1)
    plt.ylabel(vola_col2)
    plt.title(f'Volatility Clusters (K={optimal_k})')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.show()

    df_cluster = df.copy()
    df_cluster['cluster'] = clusters

    #st
    plt.figure(figsize=(12, 6))
    plt.scatter(df_cluster[time_col], df_cluster[vola_col1], c=df_cluster['cluster'], cmap='viridis', alpha=0.6)
    plt.xlabel('Time')
    plt.ylabel(vola_col1)
    plt.title('Cluster Distribution Over Time')
    plt.colorbar(label='Cluster')
    plt.xticks(rotation=45)
    plt.show()

    #lt
    plt.figure(figsize=(12, 6))
    plt.scatter(df_cluster[time_col], df_cluster[vola_col2], c=df_cluster['cluster'], cmap='viridis', alpha=0.6)
    plt.xlabel('Time')
    plt.ylabel(vola_col2)
    plt.title('Cluster Distribution Over Time')
    plt.colorbar(label='Cluster')
    plt.xticks(rotation=45)
    plt.show()

    df_cluster['cluster_shift'] = df_cluster['cluster'].shift()
    same_cluster_streak = (df_cluster['cluster'] == df_cluster['cluster_shift']).cumsum()
    streak_lengths = same_cluster_streak.value_counts().value_counts().sort_index()
    print("Consecutive appearance count:\n", streak_lengths)
    df_cluster.drop(columns='cluster_shift', inplace=True)
    return df_cluster

#df=loading('clustered_pvlabeled_aggregated_data.csv')
#df2=loading_fg('fear_greed_hourly.csv')
df3=loading('cleaned_tt.csv')

#df=calculate_avg_change(df)
#df2=calculate_avg_change_fg(df2)
df3=calculate_avg_change_tt(df3)

#df=clusters_1(df,'short_term_change','st_index')
#0--rise 1--fall 2--still
#df=clusters_1(df,'long_term_change','lt_index')
#0--rise 1--still 2--fall
#correlation = df['short_term_change'].corr(df['long_term_change'])

#print('st-lt correlation: ',correlation)

#df2=clusters_1(df2,'change','index',time_col='Timestamp')
#0--still 1--rise 2--fall

df3=clusters_1(df3,'avg_senti','index')
#0--rise 1--fall 2--still

#df.to_csv('clustered_finance.csv')
#df2.to_csv('clustered_fg.csv')
df3.to_csv('clustered_tt.csv',index=False)