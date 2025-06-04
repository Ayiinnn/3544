import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_by_hour(df, a, b):

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.floor('H')
    grouped = df.groupby('timestamp')

    result = grouped['sentiment'].agg(
        count_total='count',
        count_above=lambda x: (x > a).sum(),
        count_below=lambda x: (x < b).sum()
    ).reset_index()

    result['extreme_positive_%'] = result['count_above'] / result['count_total']
    result['extreme_negative_%'] = result['count_below'] / result['count_total']
    final_result = result[['timestamp', 'extreme_positive_%', 'extreme_negative_%']]
    return final_result

df = pd.read_csv('final2.csv')

percentile_90 = df['sentiment'].quantile(0.90)
percentile_10 = df['sentiment'].quantile(0.10)

#plt.figure(figsize=(10, 6))
#sns.histplot(df['sentiment'], bins=50, kde=True)
#plt.axvline(percentile_95, color='red', linestyle='--', label='90th Percentile')
#plt.axvline(percentile_5, color='blue', linestyle='--', label='10th Percentile')
#plt.legend()
#plt.title('Sentiment Distribution')
#plt.xlabel('Sentiment')
#plt.ylabel('Frequency')
#plt.show()

print(f'90th Percentile: {percentile_90}')
print(f'10th Percentile: {percentile_10}')

#pos：0.4430330201854541  -95
#neg：-0.3060669498134744  -5

df2=aggregate_by_hour(df,0.3592885368919117, -0.20968)
df2.to_csv('extreme_sentiment_proportion.csv',index=False)

#plt.plot(df2['timestamp'], df2['extreme_positive_%'], label='Extreme Positive %', color='green')
#plt.plot(df2['timestamp'], df2['extreme_negative_%'], label='Extreme Negative %', color='red')
#plt.title('Extreme Positive and Negative Sentiment Over Time')
#plt.xlabel('Timestamp')
#plt.ylabel('Percentage (%)')
#plt.legend()
#plt.show()
