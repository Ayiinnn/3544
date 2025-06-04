import pandas as pd
import re

def process_time_and_columns(df):
    processed_df = df.copy()
    processed_df['timestamp'] = pd.to_datetime(
        processed_df['timestamp'],
        utc=True,
        errors='coerce'
    )
    processed_df = processed_df.dropna(subset=['timestamp'])
    processed_df['timestamp'] = processed_df['timestamp'].dt.floor('H')
    processed_df = processed_df.drop(columns=[df.columns[1]])
    return processed_df

def clean(df):
    def soft_clean(text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s.,!?]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['soft_cleaned_text'] = df['content'].apply(soft_clean)
    df.drop(columns=['content'], inplace=True)
    return df

#df=pd.read_csv('cleaned_2021.csv')
#print(df.columns)
#df=process_time_and_columns(df)
#df=clean(df)
#df.to_csv('double_cleaned_2021.csv',index=False)

df2=pd.read_csv('cleaned_2022.csv')
print(df2.columns)
df2=process_time_and_columns(df2)
df2=clean(df2)
df2.to_csv('double_cleaned_2022.csv',index=False)
