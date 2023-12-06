import pandas as pd
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv('sorted.csv')

# Discard countries with fewer than 32 images
country_counts = df['country'].value_counts()
countries_to_keep = country_counts[country_counts >= 32].index
df = df[df['country'].isin(countries_to_keep)]

# Limit images for countries with more than 20,000 images
def limit_country_images(data, max_images=20000):
    return data.groupby('country').apply(lambda x: x.sample(min(len(x), max_images), random_state=42)).reset_index(drop=True)

df = limit_country_images(df)

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Split the data
train, remaining = train_test_split(df, train_size=train_ratio, random_state=42)
val, test = train_test_split(remaining, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

# Function to check coverage for a given column
def check_coverage(data_split, df, column):
    unique_values = df[column].unique()
    return all(value in data_split[column].unique() for value in unique_values)

# Validate coverage for each attribute in each split
attributes = ['country', 'category', 'continent']
for split_name, split_df in [('train', train), ('validation', val), ('test', test)]:
    for attribute in attributes:
        coverage = check_coverage(split_df, df, attribute)
        print(f"{split_name.capitalize()} split coverage for {attribute}: {coverage}")

# Save DataFrames to CSV if desired
train.to_csv('train_align.csv', index=False)
val.to_csv('validation_align.csv', index=False)
test.to_csv('test_align.csv', index=False)
