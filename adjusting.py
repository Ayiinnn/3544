import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv('final.csv')
plt.figure(figsize=(10, 6))
sns.histplot(df['vader_sentiment'], bins=50, kde=True)
plt.show()
print(df['vader_sentiment'].max(),df['vader_sentiment'].min())
plt.figure(figsize=(10, 6))

df['afinn_sentiment'] = np.sign(df['afinn_sentiment']) * (abs(df['afinn_sentiment']) ** (1/2))
df['afinn_sentiment']  = 2*(df['afinn_sentiment']/(df['afinn_sentiment'].max()-df['afinn_sentiment'].min()))

sns.histplot(df['afinn_sentiment'], bins=50, kde=True)
plt.show()
print(df['afinn_sentiment'].max(),df['afinn_sentiment'].min())
df['sentiment'] = df['vader_sentiment'] * 0.4 + df['afinn_sentiment'] * 0.6
df.to_csv('final2.csv',index=False)