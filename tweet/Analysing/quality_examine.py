import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def loading(file_path):
     df0 = pd.read_csv(file_path, parse_dates=['timestamp'])
     df0 = df0.sort_values('timestamp').reset_index(drop=True)
     return df0

def check_miss(x):
    missing_count = x.isnull().sum()
    print(missing_count)
    return 0

def check_repi(x,move):
    duplicates = x[x.duplicated('timestamp', keep=False)]
    print(len(duplicates))
    plt.figure(figsize=(10, 3))
    plt.plot(x['timestamp'], [0] * len(x), '|', label='all timestamps', color='lightgray')
    plt.scatter(duplicates['timestamp'], np.zeros(len(duplicates)), marker='|', s=100)
    plt.title(duplicates)
    plt.show()
    if move:
         x = x.drop_duplicates('timestamp', keep='last')
    return x


def check_irretime(x):
    time_diff = x['timestamp'].diff().dt.total_seconds()
    irregular_time = time_diff[(time_diff != time_diff.mode()[0]) & time_diff.notna()]

    # testing
    #for idx in irregular_time.index[:4]:  # 前四个问题行
        #error_row = x.loc[idx]
        #context_rows = x.iloc[max(0, idx - 1):min(len(x), idx + 2)]
        #print(f"abnormal line context: {idx}:\n{context_rows}\n")

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(x['timestamp'], [0] * len(x), '|', label='all timestamps', color='lightgray')
    plt.scatter(x.loc[irregular_time.index, 'timestamp'], [0] * len(irregular_time),
                color='red', label='abnormal timestamps', s=200, marker='|')
    plt.title("irregular timestamp", fontsize=16)
    plt.xlabel("time", fontsize=12)
    plt.yticks([])
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return 0


def recover_irretime(df):
    df = df.copy()
    df['flag'] = 1


    time_diff = df['timestamp'].diff().dt.total_seconds()
    mode_value = time_diff.mode()[0]
    irregular_time = time_diff[(time_diff != mode_value) & (~time_diff.isna())]

    new_rows = []
    for idx in irregular_time.index:
        prev_idx = idx - 1
        if prev_idx < 0:
            continue

        prev_time = df.loc[prev_idx, 'timestamp']
        current_time = df.loc[idx, 'timestamp']
        time_gap = current_time - prev_time


        if time_gap <= pd.Timedelta(hours=1):
            continue

        start_hour = prev_time.floor('H') + pd.Timedelta(hours=1)

        current_hour = start_hour
        while current_hour < current_time:
            new_row = df.loc[prev_idx].copy()
            new_row['timestamp'] = current_hour
            new_row['flag'] = 0
            new_rows.append(new_row)
            current_hour += pd.Timedelta(hours=1)

    if new_rows:
        df_filled = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_filled = df.copy()

    return df_filled.sort_values('timestamp').reset_index(drop=True)

df=loading("aggregated_tweets.csv")
check_miss(df)
df=check_repi(df,True)

check_irretime(df)
df2=recover_irretime(df)
check_irretime(df2)

#导出
df2.to_csv('cleaned_tt.csv', index=False)
print('FINISHED')
#可视化
#plt.plot(df2['timestamp'],df2['high'],color='red',label='high')
#plt.plot(df2['timestamp'],df2['low'],color='blue',label='low')

#plt.show()