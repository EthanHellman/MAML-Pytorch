import os
import json
import pandas as pd
import pycountry_convert as pc
import pycountry
from countryinfo import CountryInfo

df = pd.DataFrame(columns=['file_path', 'country', 'continent', 'ns-hemisphere', 'ew-hemisphere', 'category'])
img_dirs = ['/deep/group/aicc-bootcamp/self-sup/fmow_rgb/train', '/deep/group/aicc-bootcamp/self-sup/fmow_rgb/val']
all_directories = []
for img_dir in img_dirs:
    directories = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    all_directories.extend(directories)

def get_country_code(file_path):
    # Extract the
    try:
        # Open the JSON file
        with open(file_path, 'r') as json_file:
            # Load the JSON data
            data = json.load(json_file)
            
            # Extract the value associated with the 'country_code' key
            country_code = data.get('country_code', None)
            
            if country_code:
                # print(f"The country code is: {country_code}")
                return country_code
            else:
                print("The 'country_code' key does not exist in the JSON data or it has a null value.")
                return None
    except FileNotFoundError:
        print(f"The file at path {file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"The file at path {file_path} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# def country_to_continent(country_code):
#     try:
#         # Get the country name from the country code
#         country_name = pycountry.countries.get(alpha_3=country_code).name
        
#         # Convert the country name into a continent code
#         continent_code = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country_name))
        
#         # Get the continent name from the continent code
#         continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        
#         return continent_name
    
#     except Exception as e:
#         print(f"An error occurred for country code {country_code}: {e}")
#         return None

def country_to_hemishere(code):
    # for code in country_queries:
    try:
        country = pycountry.countries.get(alpha_3=code)
        country_name = country.name
        # if country_name has a ',' in it, then take the first part of the string before the comma
        if ',' in country_name:
            country_name = country_name.split(',')[0]

        continent_code = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country_name))
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)

        if country and continent_name:
            country_info = CountryInfo(country.name)
            lat = country_info.latlng()[0]
            long = country_info.latlng()[1]
            ns_hemisphere = 'Northern' if lat >= 0 else 'Southern'
            ew_hemisphere = 'Eastern' if long >= 0 else 'Western'
            # print(f'{code} ({country.name}) is in the {hemisphere} Hemisphere')
            return country.name, continent_name, ns_hemisphere, ew_hemisphere
        else:
            print(f"Country not found for code {code}")
            return None, None, None, None
    except Exception as e:
        print(f"Could not retrieve data for {code}: {str(e)}")
        return None, None, None, None

# for i in range(50):
#     dr = all_directories[i]
for img_dir in img_dirs:

    # for dr in all_directories:
    for dr in os.listdir(img_dir):

        path = os.path.join(img_dir, str(dr))
        # print(path)

        for sub_dir in os.listdir(path):
            sub_path = os.path.join(path, sub_dir)
            # print(sub_path)

            for f in os.listdir(sub_path):
                # print(f)

                if f.endswith('.jpg') and (not 'msrgb' in f):

                    jpg_path = os.path.join(sub_path, f)
                    print(jpg_path)
                    json_file = f.rsplit('.', 1)[0] + '.json'
                    json_path = os.path.join(sub_path, json_file)
                    country_code = get_country_code(json_path)
                    
                    if os.path.exists(json_path) and country_code:

                        country_name, continent_name, ns_hemi, ew_hemi = country_to_hemishere(country_code)

                        if country_name != None:
                            columns=['file_path', 'country', 'continent', 'ns-hemisphere', 'ew-hemisphere', 'category']
                            new_row = pd.DataFrame([[jpg_path, country_name, continent_name, ns_hemi, ew_hemi, dr]], columns=columns)
                            df = pd.concat([df, new_row], ignore_index=True) #df.append(new_row, ignore_index=True)

print(df)

# df = pd.DataFrame(list(country_queries.items()), columns=['Country', 'Count', 'NS Hemisphere', 'EW Hemisphere'])

df.to_csv('final.csv', index=False)
