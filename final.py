import pandas as pd

df1=pd.read_csv('aggregated_tweets.csv')
df2=pd.read_csv('extreme_sentiment_proportion.csv')
df4=pd.read_csv('fffin.csv')

print("df1的列名：", df1.columns.tolist())
print("df2的列名：", df2.columns.tolist())
print("df2的列名：", df4.columns.tolist())
print("df2的列名：",len(df4.columns.tolist()))

# 定义需要保留的列（按实际需求修改）
selected_cols_df1 = ['timestamp', 'tweets_count', 'max_senti', 'min_senti', 'avg_senti']  # 第一个DF需要保留的列
selected_cols_df2 = ['extreme_positive_%', 'extreme_negative_%']


# 创建合并后的DF
df3 = pd.concat([
    df1[selected_cols_df1],
    df2[selected_cols_df2]
], axis=1)

df3.to_csv('ffmedia.csv',index=False)
df4.drop(columns= ['index', 'typical_price_first'])
df4.to_csv('ffffin.csv',index=False)