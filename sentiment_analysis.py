import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn

def add_vader_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df['soft_cleaned_text'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )
    return df

def add_afinn_sentiment(df):
    afinn = Afinn()
    df['afinn_sentiment'] = df['soft_cleaned_text'].apply(
        lambda x: afinn.score(x) if isinstance(x, str) else 0
    )
    return df

df = pd.read_csv('double_cleaned_2021.csv')
df = add_vader_sentiment(df)
print('Vadar finished')
df = add_afinn_sentiment(df)
df.to_csv('analysed_2021.csv', index=False)