import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

excel_file_path = 'C:/Users/User/Documents/Dissertation Research/Data/Companies will be usedFINAL - Copy.xlsx'

sheet_names_map = {
    'AstraZeneca': 'AZN LN Equity',
    'Shell': 'SHEL LN Equity',
    'BHP': 'BHP AU Equity',
    'CSL': 'CSL AU Equity',
    'DBS': 'DBS SP Equity',
    'Singtel': 'ST SP Equity'
}
traditional_variables = [
    'BEST_PE_RATIO', 'CUR_MKT_CAP', 'PX_LAST', 'RSI_14D',

    'VWAP_NUM_TRADES', 'VWAP_STANDARD_DEV', 'VOLATILITY_30D', 
    'MOV_AVG_200D', 'MOV_AVG_50D', 'AVERAGE_BID_ASK_SPREAD_%',
    'Index_Volume', 'VOLATILITY_360D', 'Index_RSI 14 Day',
    'Index_Volatility 30 Day',	'Index_Volatility 360 Day', 'PE_RATIO'
]
sentiment_variables = [
    'NEWS_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_AVG',
    'NEWS_NEG_SENTIMENT_COUNT', 'TWITTER_POS_SENTIMENT_COUNT',
    'TWITTER_NEG_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT',
    'NEWS_NEUTRAL_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT'
]
google_trends_variables = [
    'companyname_rescaled',	'stockticker_rescaled',	'companystock_rescaled',
	'stocks_rescaled',	'portfolio_rescaled',	'inflation_rescaled',	
    'risk_rescaled',	'dividend_rescaled',	'journal_rescaled',	'value_rescaled',
	'finance_rescaled',	'capital_rescaled',	'option_rescaled'
]

print("Loading and combining data from Excel sheets")
all_data_list = []
try:
    all_sheets = pd.read_excel(excel_file_path, sheet_name=list(sheet_names_map.values()))
    for company_name, sheet_name in sheet_names_map.items():
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
print("\nAll data loaded and combined successfully")

# Data Cleaning 
print("\n--- 3. Cleaning Data ---")
all_vars = traditional_variables + sentiment_variables + google_trends_variables
df_analysis = df[['Date', 'Company'] + [col for col in all_vars if col in df.columns]].copy()

# Apply cleaning strategies
df_analysis[traditional_variables] = df_analysis.groupby('Company')[traditional_variables].transform(lambda x: x.bfill().ffill())
df_analysis[sentiment_variables] = df_analysis[sentiment_variables].fillna(0)
df_analysis[google_trends_variables] = df_analysis[google_trends_variables].fillna(0)
df_analysis.dropna(inplace=True)
print("Missing values handled.")

# Descriptive Statistics
print("\n" + "="*50)
print("--- 4. Descriptive Statistics for All Variables ---")
print("="*50 + "\n")

# Calculate descriptive statistics and transpose for better readability
descriptive_stats = df_analysis.describe().T
print(descriptive_stats)

# Correlation and Multicollinearity Analysis 
print("\n" + "="*50)
print("--- 5. Correlation and Multicollinearity Analysis ---")
print("="*50 + "\n")

# Correlation Matrix Heatmap 
print("Correlation Matrix Heatmap")

numerical_df = df_analysis.select_dtypes(include=np.number)
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(24, 20))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=.5, annot_kws={"size": 8})
plt.title('Overall Correlation Matrix of All Numerical Variables', fontsize=22)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Variance Inflation Factor (VIF) for Multicollinearity 
print("Calculating Variance Inflation Factor (VIF)")
print("VIF measures how much a variable is explained by other variables.")
print("A high VIF (often > 5 or 10) indicates high multicollinearity.")


# calculate VIF on the cleaned numerical data
X_vif = add_constant(numerical_df.drop(columns=['PX_VOLUME']))

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

vif_data = vif_data.sort_values(by="VIF", ascending=False)
print(vif_data)


# Time series visualization for each variable 
print("\n" + "="*50)
print("="*50 + "\n")

# Combine all variables into one list for plotting
all_variables_to_plot = traditional_variables + sentiment_variables + google_trends_variables

plt.style.use('seaborn-v0_8-whitegrid')

for var in all_variables_to_plot:
    if var in df_analysis.columns:
        print(f"Plotting {var}...")
        plt.figure(figsize=(16, 8))
        
        sns.lineplot(data=df_analysis, x='Date', y=var, hue='Company', palette='viridis')
        
        plt.title(f'Time Series of {var}', fontsize=18)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(var, fontsize=12)
        plt.legend(title='Company')
        plt.tight_layout()
        plt.show()

print("\nEDA script finished.")

