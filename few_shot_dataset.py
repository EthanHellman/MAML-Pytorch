import random
from torch.utils.data import Subset

def precompute_country_indices(grouped_df):
    print("Pre-computing country indices from grouped DataFrame")
    country_indices = {name: list(group.index) for name, group in grouped_df}
    return country_indices

def create_few_shot_dataset(dataset, num_shot, num_way, country_indices):
    # print("Creating few-shot dataset")
    few_shot_dataset_indices = []

    # Randomly select 'num_way' countries
    selected_countries = random.sample(list(country_indices.keys()), num_way)

    # For each selected country, pick 'num_shot' samples
    for country in selected_countries:
        country_indices_list = country_indices[country]
        if len(country_indices_list) >= num_shot:
            few_shot_dataset_indices.extend(random.sample(country_indices_list, num_shot))
        else:
            few_shot_dataset_indices.extend(country_indices_list)

    return Subset(dataset, few_shot_dataset_indices)
