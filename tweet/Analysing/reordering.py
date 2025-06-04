import pandas as pd

df=pd.read_csv('final.csv')
def transform_columns(df):
    df.iloc[:, 1] = df.iloc[:, 1].replace({0: 1, 1: -1, 2: 0})
    df.iloc[:, 2] = df.iloc[:, 2].replace({0: 1, 1: 0, 2: -1})
    df.iloc[:, 3] = df.iloc[:, 3].replace({2: -1})
    df.iloc[:, 4] = df.iloc[:, 4].replace({0: 1, 1: -1, 2: 0})
    return df
df=transform_columns(df)
df.to_csv('final2.csv',index=False)