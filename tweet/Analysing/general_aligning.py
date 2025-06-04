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

df=loading('clustered_finance.csv')
df2=loading_fg('clustered_fg.csv')
df3=loading('clustered_tt.csv')
df6=loading('finalfin.csv')
df6_s=df6.iloc[29588:42692].reset_index()


df6_s.to_csv('fffin.csv',index=False)


df_s=df.iloc[29588:42692].reset_index()
df_s=df_s[['timestamp','st_index', 'lt_index']]
print(df_s)
df2_s=(df2.iloc[25560:38664]).reset_index()
df2_s=df2_s[['Timestamp','index']]
print(df2_s)
df3_s=df3[['timestamp','index']].reset_index()
print(df3_s)

df_s2=df.iloc[29588:42692].reset_index()
df_s2=df_s2[['timestamp','close','vola_composite_index','lt_volatility','long_term_change','short_term_change','volume']]
print(df_s2)
df2_s2=(df2.iloc[25560:38664]).reset_index()
df2_s2=df2_s2[['Value']]
print(df2_s)
df3_s2=df3[['avg_senti']].reset_index()
print(df3_s2)

df4=pd.DataFrame()
df4['timestamp']=df_s['timestamp']
df4['fst_index']=df_s['st_index']
df4['flt_index']=df_s['lt_index']
df4['fg_index']=df2_s['index']
df4['tt_index']=df3_s['index']
df4.to_csv('final.csv',index=False)

df5=pd.DataFrame()
df5['timestamp']=df_s2['timestamp']
df5['finance1']=df_s2['close']
df5['finance2']=df_s2['vola_composite_index']
df5['finance3']=df_s2['lt_volatility']
df5['finance4']=df_s2['short_term_change']
df5['finance5']=df_s2['long_term_change']
df5['finance6']=df_s2['volume']
df5['fg']=df2_s2['Value']
df5['tt']=df3_s2['avg_senti']
df5.to_csv('relation.csv',index=False)