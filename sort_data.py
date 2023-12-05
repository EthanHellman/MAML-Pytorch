import pandas as pd

df = pd.read_csv('final.csv')

df_2 = pd.read_csv('sorted.csv')

unique_countries = df_2["country"].unique()

print(unique_countries)