import pandas as pd
import sys

excel_file_path = 'C:/Users/User/Documents/Dissertation Research/Data/Companies will be usedFINAL - Copy.xlsx'

sheet_names_map = {
    'AstraZeneca': ('AZN LN Equity', 'AZN'),
    'Shell': ('SHEL LN Equity', 'SHEL'),
    'BHP': ('BHP AU Equity', 'BHP'),
    'CSL': ('CSL AU Equity', 'CSL'),
    'DBS': ('DBS SP Equity', 'D05'),
    'Singtel': ('ST SP Equity', 'Z74')}

volume_variable = 'PX_VOLUME'

# Load and combine data from the single Excel File
print("--- 1. Loading and Combining Data from Excel Sheets ---")
all_data_list = []
try:
    #extracts only the sheet name
    all_sheets = pd.read_excel(excel_file_path, sheet_name=[v[0] for v in sheet_names_map.values()])
    for company_name, (sheet_name, ticker) in sheet_names_map.items():
        company_df = all_sheets[sheet_name]
        company_df['Company'] = company_name
        all_data_list.append(company_df)
except FileNotFoundError:
    print(f"Error: The file was not found at '{excel_file_path}'. Please check the path.")
    sys.exit()
except Exception as e:
    print(f"An error occurred while loading the Excel file: {e}")
    sys.exit()

df = pd.concat(all_data_list, ignore_index=True)
df.rename(columns={'Dates': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Company', 'Date']).reset_index(drop=True)
print("Data loaded successfully.")

# Create the target variable
print("\n--- 2. Engineering the Target Variable ---")
# Create a copy to avoid modifying the original dataframe during calculations
df_target = df[['Date', 'Company', volume_variable]].copy()

# no missing volume numbers before calculating the difference
df_target.dropna(subset=[volume_variable], inplace=True)

# Calculate the change in volume from the previous day
df_target['volume_change'] = df_target.groupby('Company')[volume_variable].diff()

# Create the binary target: 1 if volume increased, 0 otherwise
df_target['volume_direction'] = (df_target['volume_change'] > 0).astype(int)

# Drop the first row for each company which will have a NaN for volume_change
df_target.dropna(subset=['volume_change'], inplace=True)
print("Target variable 'volume_direction' created.")

# Calculate and show class balance
print("\n" + "="*50)
print("--- 3. Class Balance Analysis ---")
print("="*50 + "\n")

# Loop through each company to calculate and print the balance
for company in sheet_names_map.keys():
    company_data = df_target[df_target['Company'] == company]
    
    if company_data.empty:
        print(f"No data available for {company}.")
        continue
        
    print(f"--- Class Balance for: {company} ---")
    
    # Use value_counts to get the counts of each class
    counts = company_data['volume_direction'].value_counts()
    
    # Use value_counts with normalize=True to get the percentages
    percentages = company_data['volume_direction'].value_counts(normalize=True) * 100
    
    decrease_count = counts.get(0, 0)
    increase_count = counts.get(1, 0)
    
    decrease_percent = percentages.get(0, 0)
    increase_percent = percentages.get(1, 0)
    print(f"  - Decrease/Same (Class 0): {decrease_count} instances ({decrease_percent:.2f}%)")
    print(f"  - Increase (Class 1)     : {increase_count} instances ({increase_percent:.2f}%)")
    print("-" * 35)

