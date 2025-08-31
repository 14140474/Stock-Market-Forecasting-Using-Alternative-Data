import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlxtend.evaluate import mcnemar_table, mcnemar
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

traditional_variables = [
     'BEST_PE_RATIO',
     'VWAP_NUM_TRADES', 'VWAP_STANDARD_DEV', 'VOLATILITY_30D',  
     'Index_Volume']
sentiment_avg_variables = ['NEWS_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_AVG']
sentiment_count_variables = [
    'TWITTER_POS_SENTIMENT_COUNT',
   'TWITTER_NEG_SENTIMENT_COUNT']
google_trends_variables = [
    'companyname_rescaled',	'stockticker_rescaled',	'companystock_rescaled',
	'stocks_rescaled',	'portfolio_rescaled',	'inflation_rescaled',	
    'risk_rescaled',	'dividend_rescaled',	'journal_rescaled',	'value_rescaled',
	'finance_rescaled',	'capital_rescaled',	'option_rescaled']

print(" Loading and Combining Data from Excel Sheets")
all_data_list = []
try:
    # Load all sheets from the Excel file at once, since column names are consistent
    all_sheets = pd.read_excel(excel_file_path, sheet_name=[v[0] for v in sheet_names_map.values()])
    
    for company_name, (sheet_name, ticker) in sheet_names_map.items():
        company_df = all_sheets[sheet_name]
        company_df['Company'] = company_name
        company_df['Ticker'] = ticker
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
print("\nAll data loaded and combined successfully!")


#  Data Cleaning
print(" Cleaning Data")
all_sentiment_vars = sentiment_avg_variables + sentiment_count_variables
all_vars = traditional_variables + all_sentiment_vars + google_trends_variables
df_analysis = df[['Date', 'Company', 'Ticker', volume_variable] + [col for col in all_vars if col in df.columns]].copy()

# Apply cleaning strategies
df_analysis[traditional_variables] = df_analysis.groupby('Company')[traditional_variables].transform(lambda x: x.bfill().ffill())
df_analysis[[col for col in all_sentiment_vars if col in df.columns]] = df_analysis[[col for col in all_sentiment_vars if col in df.columns]].fillna(0)
df_analysis[[col for col in google_trends_variables if col in df.columns]] = df_analysis[[col for col in google_trends_variables if col in df.columns]].fillna(0)
df_analysis.dropna(inplace=True)
print("Missing values handled.")


#  Feature Engineering
print(" Engineering Features for Modeling")
df_model_ready = df_analysis.copy()

df_model_ready['NEWS_SENTIMENT_MAGNITUDE'] = df_model_ready['NEWS_SENTIMENT_DAILY_AVG'].abs()
df_model_ready['TWITTER_SENTIMENT_MAGNITUDE'] = df_model_ready['TWITTER_SENTIMENT_DAILY_AVG'].abs()
sentiment_magnitude_variables = ['NEWS_SENTIMENT_MAGNITUDE', 'TWITTER_SENTIMENT_MAGNITUDE']
sentiment_vars_for_model = sentiment_avg_variables + sentiment_count_variables + sentiment_magnitude_variables

df_model_ready['volume_direction'] = (df_model_ready.groupby('Company')[volume_variable].diff() > 0).astype(int)

#  Only create lagged features for variables not available before market open
lagged_features_to_create = [volume_variable] + traditional_variables + google_trends_variables
lags = [1, 2, 3] 
for feature in lagged_features_to_create:
    if feature in df_model_ready.columns:
        for i in lags:
            df_model_ready[f'{feature}_lag_{i}'] = df_model_ready.groupby('Company')[feature].shift(i)

df_model_ready.dropna(inplace=True)
print("Lagged features created. Sentiment variables will be used in their original (non-lagged) form.")


# Independent Model Training for Each Company 
print("\n" + "="*50)
print("Training All Models for Each Company ")
print("="*50 + "\n")

all_model_results = []
all_predictions = {} 

for company_name, (sheet_name, ticker) in sheet_names_map.items():
    print(" Processing Models for: {company_name} ")
    company_data = df_model_ready[df_model_ready['Company'] == company_name].copy()
    if company_data.empty: continue

    split_date = '2024-01-01'
    train_df = company_data[company_data['Date'] < split_date]
    test_df = company_data[company_data['Date'] >= split_date]
    if train_df.empty or test_df.empty: continue
    
    y_test = test_df['volume_direction']
    company_predictions = {'actuals': y_test}

    #  Define Feature Sets with non-lagged sentiment 
    base_features = [f'{var}_lag_{i}' for var in [volume_variable] + traditional_variables for i in lags]
    attention_features = base_features + [f'{var}_lag_{i}' for var in google_trends_variables for i in lags]
    # Sentiment and Hybrid models use non-lagged sentiment variables
    sentiment_features = base_features + sentiment_vars_for_model
    hybrid_features = attention_features + sentiment_vars_for_model
    
    models_to_run = {
        "Baseline": base_features, "Attention": attention_features,
        "Sentiment": sentiment_features, "Hybrid": hybrid_features }

    for model_name, features in models_to_run.items():
        print(f"\n-- Training {model_name} model for {company_name} --")
        
        features = [f for f in features if f in train_df.columns]
        
        X_train, y_train = train_df[features], train_df['volume_direction']
        X_test = test_df[features]

        model = RandomForestClassifier(
            n_estimators=500, # Using a larger number of trees for better performance
            random_state=42,
            n_jobs=-1) # Use all available CPU cores to speed up training

        
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        company_predictions[model_name] = predictions
        
        report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        
        result_row = {
            'Company': company_name, 'Model': model_name, 'Accuracy': report_dict['accuracy'],
            'F1_Macro_Avg': report_dict['macro avg']['f1-score'],
            'F1_Weighted_Avg': report_dict['weighted avg']['f1-score'],
            'F1_Decrease_Same': report_dict.get('0', {}).get('f1-score', 0),
            'F1_Increase': report_dict.get('1', {}).get('f1-score', 0)}
        all_model_results.append(result_row)

    all_predictions[company_name] = company_predictions
    print("-" * 50)

# Statistical Significance Testing 
print("\n" + "="*50)
print("--- 6. Statistical Significance Testing ---")
print("="*50 + "\n")

statistical_results = []
for company_name, predictions_dict in all_predictions.items():
    y_target = predictions_dict['actuals']
    
    comparisons = [
        ("Hybrid", "Baseline"), 
        ("Hybrid", "Attention"), 
        ("Hybrid", "Sentiment"),
        ("Attention", "Baseline"),
        ("Sentiment", "Baseline"),
        ("Attention", "Sentiment")]
    for model2_name, model1_name in comparisons:
        if model1_name in predictions_dict and model2_name in predictions_dict:
            preds1 = predictions_dict[model1_name]
            preds2 = predictions_dict[model2_name]
            
            tb = mcnemar_table(y_target=y_target, y_model1=preds1, y_model2=preds2)
            chi2, p_value = mcnemar(ary=tb, corrected=True)
            
            stat_row = {'Company': company_name, 'Comparison': f'{model2_name}_vs_{model1_name}', 'p_value': p_value}
            statistical_results.append(stat_row)
            print(f"McNemar's Test for {company_name} ({model2_name} vs {model1_name}): p-value = {p_value:.4f}")

# save results
if all_model_results:
    results_summary_df = pd.DataFrame(all_model_results)
    
    if statistical_results:
        stats_summary_df = pd.DataFrame(statistical_results)
        stats_pivot_df = stats_summary_df.pivot(index='Company', columns='Comparison', values='p_value').reset_index()
        final_summary_df = pd.merge(results_summary_df, stats_pivot_df, on='Company', how='left')
    else:
        final_summary_df = results_summary_df

    final_filename = 'final_summary_and_stats_randomforest2.csv'
    final_summary_df.to_csv(final_filename, index=False)
    print(f"\nAll model performance and statistical test results saved to '{final_filename}'")
    print("\n--- Final Summary Table ---")
    print(final_summary_df)

