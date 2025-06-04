import pandas as pd
from wtforms.utils import clean_datetime_format_for_strptime


def remove_soft_cleaned_column(df):
    return df.drop(columns=['soft_cleaned_text'])


def clean_dataframe(df):
    initial_count = len(df)
    df_cleaned = df[~((df['soft_cleaned_text'].str.len() < 23) &
                      ((df['vader_sentiment'] == 0) | (df['afinn_sentiment'] == 0)))]

    deleted_count = initial_count - len(df_cleaned)
    return df_cleaned, initial_count, deleted_count

df1 = pd.read_csv('analysed_2021.csv', compression='gzip')
df2 = pd.read_csv('analysed_2022.csv', compression='gzip')

df1_clean,count_11, count_12= clean_dataframe(df1)
df1_clean = remove_soft_cleaned_column(df1)

print(count_11,count_12)

df2_clean,count_21, count_22= clean_dataframe(df2)
df2_clean = remove_soft_cleaned_column(df2)

print(count_21,count_22)
df3=pd.concat([df1_clean, df2_clean], axis=0)
df3.to_csv('final.csv')