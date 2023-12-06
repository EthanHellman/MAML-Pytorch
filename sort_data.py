import pandas as pd

# Load your data
df = pd.read_csv('sorted.csv')

# Create a dictionary for country to category mapping
country_to_category = df.groupby('country')['category'].first().to_dict()

# Group by country and count the number of images
image_counts_series = df.groupby('country').size()
image_counts = {country: min(count, 20000) for country, count in image_counts_series.items() if count >= 32}
# image_counts = {country: count for country, count in image_counts_series.items() if count >= 128}

# Print total number of images
print(f"Total number of images: {sum(image_counts.values())}")

# Total number of countries after filtering
total_countries = len(image_counts)
print(f"Total number of countries: {total_countries}")

# Target number of countries and images for each set
train_countries_target = round(total_countries * 0.7)
val_countries_target = round(total_countries * 0.15)
test_countries_target = total_countries - train_countries_target - val_countries_target

total_images = sum(image_counts.values())
train_target = total_images * 0.7
val_target = total_images * 0.15
test_target = total_images - train_target - val_target

# Sort countries by image count in ascending order
sorted_countries_asc = sorted(image_counts.items(), key=lambda x: x[1])

# Initialize sets and counters
train_set, val_set, test_set = set(), set(), set()
train_count, val_count, test_count = 0, 0, 0
train_countries, val_countries, test_countries = 0, 0, 0

# Initialize sets to keep track of categories in each split
train_categories, val_categories, test_categories = set(), set(), set()

# Allocation function
def allocate(country, count, set_name):
    global train_count, val_count, test_count
    global train_countries, val_countries, test_countries
    global train_categories, val_categories, test_categories
    
    category = country_to_category[country]
    
    if set_name == 'train':
        train_set.add(country)
        train_count += count
        train_countries += 1
        train_categories.add(category)
    elif set_name == 'val':
        val_set.add(country)
        val_count += count
        val_countries += 1
        val_categories.add(category)
    else:  # 'test'
        test_set.add(country)
        test_count += count
        test_countries += 1
        test_categories.add(category)

# Pre-allocation to ensure each category is represented in each set
unique_categories = set(df['category'].unique())

for category in unique_categories:
    for country, count in sorted_countries_asc:
        if country_to_category[country] == category:
            if category not in train_categories:
                allocate(country, count, 'train')
                break
            elif category not in val_categories:
                allocate(country, count, 'val')
                break
            elif category not in test_categories:
                allocate(country, count, 'test')
                break

# Continue with existing weighted allocation logic
for country, count in sorted_countries_asc:
    if country not in train_set and country not in val_set and country not in test_set:
        missing_train_categories = unique_categories - train_categories
        missing_val_categories = unique_categories - val_categories
        missing_test_categories = unique_categories - test_categories

        country_category = country_to_category[country]

        # Calculate the current weight (image count and country count) for each set
        train_weight = (train_count / train_target) + (train_countries / train_countries_target)
        val_weight = (val_count / val_target) + (val_countries / val_countries_target)
        test_weight = (test_count / test_target) + (test_countries / test_countries_target)

        # Allocate based on the smallest weight and prioritizing missing categories
        if (country_category in missing_train_categories and train_weight <= val_weight and train_weight <= test_weight) or \
           (train_weight < val_weight and train_weight < test_weight):
            allocate(country, count, 'train')
        elif (country_category in missing_val_categories and val_weight <= train_weight and val_weight <= test_weight) or \
             (val_weight < train_weight and val_weight < test_weight):
            allocate(country, count, 'val')
        elif country_category in missing_test_categories or test_weight <= train_weight and test_weight <= val_weight:
            allocate(country, count, 'test')

# Verify the distribution
print(f"Train Set: {train_count} images, {len(train_set)} countries")
print(f"Validation Set: {val_count} images, {len(val_set)} countries")
print(f"Test Set: {test_count} images, {len(test_set)} countries")

# Function to check and print missing categories
def check_and_print_missing_categories(set_countries, df, set_name):
    # Filter the DataFrame to only include countries in the set
    set_df = df[df['country'].isin(set_countries)]
    
    # Get the unique categories in this set
    set_categories = set(set_df['category'].unique())

    # Determine missing categories
    missing_categories = unique_categories - set_categories

    if missing_categories:
        print(f"{set_name} Set is missing categories: {missing_categories}")
        return False
    else:
        print(f"All categories present in {set_name} Set.")
        return True

# Perform the checks and print missing categories
train_contains_all = check_and_print_missing_categories(train_set, df, "Train")
val_contains_all = check_and_print_missing_categories(val_set, df, "Validation")
test_contains_all = check_and_print_missing_categories(test_set, df, "Test")

# Print the results
print(f"All categories in Train Set: {train_contains_all}")
print(f"All categories in Validation Set: {val_contains_all}")
print(f"All categories in Test Set: {test_contains_all}")

# Check for overlap between sets
overlap_train_val = train_set.intersection(val_set)
overlap_train_test = train_set.intersection(test_set)
overlap_val_test = val_set.intersection(test_set)

# Print the results
print(f"Overlap between Train and Validation sets: {overlap_train_val}")
print(f"Overlap between Train and Test sets: {overlap_train_test}")
print(f"Overlap between Validation and Test sets: {overlap_val_test}")

# Check if any overlap exists
any_overlap = overlap_train_val or overlap_train_test or overlap_val_test
print(f"Is there any overlap between the sets? {'Yes' if any_overlap else 'No'}")

# Define the maximum image count per country
max_images_per_country = 20000

# Function to filter and limit DataFrame based on set countries and image count
def filter_and_limit(df, set_countries, max_images):
    filtered_df = df[df['country'].isin(set_countries)]
    limited_df = pd.DataFrame()

    for country in set_countries:
        country_df = filtered_df[filtered_df['country'] == country]
        if len(country_df) > max_images:
            # shuffle the DataFrame and select the first max_images
            country_df = country_df.sample(frac=1).reset_index(drop=True)
            country_df = country_df.head(max_images)
        limited_df = pd.concat([limited_df, country_df])

    return limited_df

# Apply filtering and limiting to each set
train_df = filter_and_limit(df, train_set, max_images_per_country)
val_df = filter_and_limit(df, val_set, max_images_per_country)
test_df = filter_and_limit(df, test_set, max_images_per_country)

# Function to check if a DataFrame contains all categories and print missing ones
def check_and_print_missing_categories_in_df(df, set_name, unique_categories):
    set_categories = df['category'].unique()
    missing_categories = unique_categories - set(set_categories)

    if missing_categories:
        print(f"{set_name} Set is missing categories: {missing_categories}")
    else:
        print(f"All categories present in {set_name} Set.")

# Get unique categories from the original DataFrame
unique_categories = set(df['category'].unique())

# Check each DataFrame for all categories
check_and_print_missing_categories_in_df(train_df, "Train", unique_categories)
check_and_print_missing_categories_in_df(val_df, "Validation", unique_categories)
check_and_print_missing_categories_in_df(test_df, "Test", unique_categories)

# Function to check for overlap in countries between DataFrames
def check_for_overlap(df1, df2, name1, name2):
    overlap = set(df1['country'].unique()) & set(df2['country'].unique())
    if overlap:
        print(f"Overlap between {name1} and {name2} sets: {overlap}")
    else:
        print(f"No overlap between {name1} and {name2} sets.")

# Check for overlap between the sets
check_for_overlap(train_df, val_df, "Train", "Validation")
check_for_overlap(train_df, test_df, "Train", "Test")
check_for_overlap(val_df, test_df, "Validation", "Test")

# Save to CSV files
train_df.to_csv('train_set.csv', index=False)
val_df.to_csv('val_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)