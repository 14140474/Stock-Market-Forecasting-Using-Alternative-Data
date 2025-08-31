import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlxtend.evaluate import mcnemar_table, mcnemar
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    'BEST_PE_RATIO', 'CUR_MKT_CAP', 'PX_LAST', 'RSI_14D',

    'VWAP_NUM_TRADES', 'VWAP_STANDARD_DEV', 'VOLATILITY_30D', 
    'MOV_AVG_200D', 'MOV_AVG_50D', 'AVERAGE_BID_ASK_SPREAD_%',
    'Index_Volume', 'VOLATILITY_360D', 'Index_RSI 14 Day',
    'Index_Volatility 30 Day',	'Index_Volatility 360 Day', 'PE_RATIO']
sentiment_avg_variables = ['NEWS_SENTIMENT_DAILY_AVG', 'TWITTER_SENTIMENT_DAILY_AVG']
sentiment_count_variables = [
    'NEWS_NEG_SENTIMENT_COUNT', 'TWITTER_POS_SENTIMENT_COUNT',
    'TWITTER_NEG_SENTIMENT_COUNT', 'NEWS_POS_SENTIMENT_COUNT',
    'NEWS_NEUTRAL_SENTIMENT_COUNT', 'TWITTER_NEUTRAL_SENTIMENT_CNT']
google_trends_variables = [
    'companyname_rescaled',	'stockticker_rescaled',	'companystock_rescaled',
	'stocks_rescaled',	'portfolio_rescaled',	'inflation_rescaled',	
    'risk_rescaled',	'dividend_rescaled',	'journal_rescaled',	'value_rescaled',
	'finance_rescaled',	'capital_rescaled',	'option_rescaled']

print(" Loading and Combining Data from Excel Sheets ")
all_data_list = []
try:
    all_sheets = pd.read_excel(excel_file_path, sheet_name=[v[0] for v in sheet_names_map.values()])
    for company_name, (sheet_name, ticker) in sheet_names_map.items():
        company_df = all_sheets[sheet_name]
        company_df['Company'] = company_name
        company_df['Ticker'] = ticker
        all_data_list.append(company_df)
except FileNotFoundError:
    print(f"Error: The file was not found at '{excel_file_path}'. Please check the path.")
    sys.exit()

df = pd.concat(all_data_list, ignore_index=True)
df.rename(columns={'Dates': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Company', 'Date']).reset_index(drop=True)
print("\nAll data loaded and combined successfully!")


#  Data Cleaning 
print("Cleaning Data ")
all_sentiment_vars = sentiment_avg_variables + sentiment_count_variables
all_vars = traditional_variables + all_sentiment_vars + google_trends_variables
df_analysis = df[['Date', 'Company', 'Ticker', volume_variable] + [col for col in all_vars if col in df.columns]].copy()

df_analysis[traditional_variables] = df_analysis.groupby('Company')[traditional_variables].transform(lambda x: x.bfill().ffill())
df_analysis[[col for col in all_sentiment_vars if col in df.columns]] = df_analysis[[col for col in all_sentiment_vars if col in df.columns]].fillna(0)
df_analysis[[col for col in google_trends_variables if col in df.columns]] = df_analysis[[col for col in google_trends_variables if col in df.columns]].fillna(0)
df_analysis.dropna(inplace=True)
print("Missing values handled.")


# Feature Engineering
print(" Engineering Features for Modeling ")
df_model_ready = df_analysis.copy()

# Create sentiment magnitude features
df_model_ready['NEWS_SENTIMENT_MAGNITUDE'] = df_model_ready['NEWS_SENTIMENT_DAILY_AVG'].abs()
df_model_ready['TWITTER_SENTIMENT_MAGNITUDE'] = df_model_ready['TWITTER_SENTIMENT_DAILY_AVG'].abs()
sentiment_magnitude_variables = ['NEWS_SENTIMENT_MAGNITUDE', 'TWITTER_SENTIMENT_MAGNITUDE']
sentiment_vars_for_pca = sentiment_avg_variables + sentiment_count_variables + sentiment_magnitude_variables

# Create the target variable
df_model_ready['volume_direction'] = (df_model_ready.groupby('Company')[volume_variable].diff() > 0).astype(int)

# Create lagged features for all predictors that are not known before market open
lagged_features_to_create = [volume_variable] + traditional_variables + google_trends_variables
lags = [1, 2, 3] 
for feature in lagged_features_to_create:
    if feature in df_model_ready.columns:
        for i in lags:
            df_model_ready[f'{feature}_lag_{i}'] = df_model_ready.groupby('Company')[feature].shift(i)

df_model_ready.dropna(inplace=True)
print("Lagged features and classification target created.")


#  Independent Model Training for Each Company -
print(" Training All Models for Each Company ")
print("="*50 + "\n")

all_model_results = []
all_predictions = {} 

for company_name, (sheet_name, ticker) in sheet_names_map.items():
    print(f"\n--- Processing Models for: {company_name} ---")
    company_data = df_model_ready[df_model_ready['Company'] == company_name].copy()
    if company_data.empty: continue

    split_date = '2024-01-01'
    train_df = company_data[company_data['Date'] < split_date].copy()
    test_df = company_data[company_data['Date'] >= split_date].copy()
    if train_df.empty or test_df.empty: continue
    
    y_train = train_df['volume_direction']
    y_test = test_df['volume_direction']
    company_predictions = {'actuals': y_test}

    # PCA Transformation (Fitted on Training data only) 
    
    # Traditional Factor
    trad_lagged_features = [f'{var}_lag_{i}' for var in traditional_variables if var != volume_variable for i in lags]
    scaler_trad = StandardScaler()
    pca_trad = PCA(n_components=1)
    train_df['traditional_factor'] = pca_trad.fit_transform(scaler_trad.fit_transform(train_df[trad_lagged_features]))
    test_df['traditional_factor'] = pca_trad.transform(scaler_trad.transform(test_df[trad_lagged_features]))

    # Sentiment Factor (using non-lagged data)
    sentiment_features_pca = [col for col in sentiment_vars_for_pca if col in train_df.columns]
    scaler_sent = StandardScaler()
    pca_sent = PCA(n_components=1)
    train_df['sentiment_factor'] = pca_sent.fit_transform(scaler_sent.fit_transform(train_df[sentiment_features_pca]))
    test_df['sentiment_factor'] = pca_sent.transform(scaler_sent.transform(test_df[sentiment_features_pca]))

    # Attention Factor
    attention_lagged_features = [f'{var}_lag_{i}' for var in google_trends_variables for i in lags]
    scaler_attn = StandardScaler()
    pca_attn = PCA(n_components=1)
    train_df['google_trends_factor'] = pca_attn.fit_transform(scaler_attn.fit_transform(train_df[attention_lagged_features]))
    test_df['google_trends_factor'] = pca_attn.transform(scaler_attn.transform(test_df[attention_lagged_features]))

    # Define Feature Sets using PCA Factors 
    base_features = [f'{volume_variable}_lag_1', f'{volume_variable}_lag_2', f'{volume_variable}_lag_3', 'traditional_factor']
    attention_features = base_features + ['google_trends_factor']
    sentiment_features = base_features + ['sentiment_factor']
    hybrid_features = base_features + ['google_trends_factor', 'sentiment_factor']
    
    models_to_run = {
        "Baseline": base_features, "Attention": attention_features,
        "Sentiment": sentiment_features, "Hybrid": hybrid_features }

    for model_name, features in models_to_run.items():
        print(f"\n-- Training {model_name} model for {company_name} --")
        
        X_train, X_test = train_df[features], test_df[features]

        # Using XGBoostClassifier 
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss')
        
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        company_predictions[model_name] = predictions
        
        report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        
        result_row = {
            'Company': company_name, 'Model': model_name, 'Accuracy': report_dict['accuracy'],
            'F1_Macro_Avg': report_dict['macro avg']['f1-score'],
            'F1_Weighted_Avg': report_dict['weighted avg']['f1-score'] }
        all_model_results.append(result_row)

    all_predictions[company_name] = company_predictions
    print("-" * 50)

#  Statistical Significance Testing 
print("\n" + "="*50)
print("Statistical Significance Testing ")
print("="*50 + "\n")

statistical_results = []
for company_name, predictions_dict in all_predictions.items():
    y_target = predictions_dict['actuals']
    
    comparisons = [
        ("Hybrid", "Baseline"), ("Hybrid", "Attention"), ("Hybrid", "Sentiment"),
        ("Attention", "Baseline"), ("Sentiment", "Baseline"), ("Attention", "Sentiment")
    ]
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

    final_filename = 'final_summary_and_stats_pca_xgboost_new.csv'
    final_summary_df.to_csv(final_filename, index=False)
    print(f"\nAll model performance and statistical test results saved to '{final_filename}'")
    print("\n--- Final Summary Table ---")
    print(final_summary_df)
