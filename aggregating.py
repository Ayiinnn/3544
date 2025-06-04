import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("final2.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'].str[:13], errors='coerce')
grouped = df.groupby('timestamp')
result = grouped.agg(
    tweets_count=('timestamp', 'size'),
    max_senti=('sentiment', 'max'),
    min_senti=('sentiment', 'min'),
    avg_senti=('sentiment', 'mean')
).reset_index()

result.to_csv("aggregated_tweets.csv", index=False)
print(result.head(7))