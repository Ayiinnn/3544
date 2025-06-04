import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math

df=pd.read_csv('aggregated_tweets.csv')
plt.plot(df['timestamp'],df["avg_senti"])
plt.show()