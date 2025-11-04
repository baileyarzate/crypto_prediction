import pandas as pd
import numpy as np
import os
import joblib

# Add the project root directory to the Python path to allow imports from 'helpers'
import sys
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
except NameError:
    project_root = os.getcwd()
sys.path.append(project_root)
# Import your custom feature engineering functions from the helpers directory
from helpers.feature_engineering import feature_engineering

# ==============================================================================
# 1. DATA PREPROCESSING FUNCTION (Copied from train.py)
# ==============================================================================
def preprocess_data(quant_path, google_path, interest_path):
    """
    This function must be IDENTICAL to the one in your training script.
    """
    print("--- Starting Data Preprocessing ---")
    df_quant = pd.read_csv(quant_path)
    df_google = pd.read_csv(google_path)
    df_interest = pd.read_csv(interest_path)

    # Normalize Dates
    df_quant['merge_date'] = pd.to_datetime(df_quant['datetime_utc'], utc=True).dt.tz_convert(None).dt.normalize()
    df_interest['merge_date'] = pd.to_datetime(df_interest['record_date']).dt.normalize()
    df_google['published'] = pd.to_datetime(df_google['published'], format="%a, %d %b %Y %H:%M:%S %Z", utc=True, errors='coerce')
    df_google.dropna(subset=['published'], inplace=True)
    df_google['merge_date'] = df_google['published'].dt.tz_convert(None).dt.normalize()

    # Aggregate/Pivot
    numeric_cols = df_google.select_dtypes(include=np.number).columns.tolist()
    df_google_agg = df_google.groupby('merge_date')[numeric_cols].mean().reset_index()
    df_interest_wide = df_interest.pivot(index='record_date', columns='security_desc', values='avg_interest_rate_amt').reset_index()
    df_interest_wide.columns = [col.lower().replace(' ', '_') for col in df_interest_wide.columns]
    
    # Align Keys for Merging
    df_quant['interest_month_key'] = df_quant['merge_date'].dt.strftime('%Y-%m')
    df_interest_wide['record_date'] = pd.to_datetime(df_interest_wide['record_date'])
    df_interest_wide['interest_month_key'] = (df_interest_wide['record_date'] + pd.DateOffset(months=1)).dt.strftime('%Y-%m')
    df_google_agg['merge_date'] = df_google_agg['merge_date'] + pd.DateOffset(days=1)

    # Merge
    df_merged = pd.merge(df_quant, df_interest_wide.drop(columns='record_date'), on='interest_month_key', how='left')
    df_final = pd.merge(df_merged, df_google_agg, on='merge_date', how='left')

    # Clean
    df_final.ffill(inplace=True)
    df_final.dropna(inplace=True)
    df_final.drop(columns=['interest_month_key'], inplace=True, errors='ignore')
    print("--- Data Preprocessing Complete ---")
    return df_final.reset_index(drop=True)

# ==============================================================================
# 2. MASTER PREDICTION FUNCTION
# ==============================================================================
def predict_next_day(df, models_dir):
    """
    Loads new raw data, processes it, loads trained models from a specific directory,
    and makes predictions for the next day.
    """
    print("--- Starting Prediction Pipeline ---")
    
    # --- 1. Load and Preprocess New Data ---
    #df = preprocess_data(quant_path, google_path, interest_path)
    
    # Ensure there's enough historical data to create features (e.g., at least 15 days)
    if len(df) < 15:
        print(f"Error: Not enough historical data ({len(df)} rows) to generate features. Need at least 15.")
        return None

    # --- 2. Load Production Artifacts ---
    try:
        print(f"Loading artifacts from: {models_dir}")
        mlr_model = joblib.load(os.path.join(models_dir, 'mlr_model.joblib'))
        ridge_model = joblib.load(os.path.join(models_dir, 'ridge_model.joblib'))
        rf_model = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
        xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.joblib'))
        #arima_model = joblib.load(os.path.join(models_dir, 'arima_model.joblib'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        features = joblib.load(os.path.join(models_dir, 'feature_list.joblib'))
    except FileNotFoundError as e:
        print(f"Error: Artifact file not found. {e}. Ensure the models_dir is correct and training was completed.")
        return None

    # --- 3. Feature Engineering ---
    df_featured = feature_engineering(df)
    
    # The last row now contains the most recent data with all features calculated
    latest_features_row = df_featured.iloc[-1]
    
    if latest_features_row[features].isnull().any():
        print("Error: Could not generate a complete feature set. NaNs detected after feature engineering.")
        return None

    # --- 4. Make Predictions ---
    
    # ML Models
    final_feature_vector = latest_features_row[features]
    final_feature_df = final_feature_vector.to_frame().T # Convert to DF to avoid feature name warning
    scaled_feature_vector = scaler.transform(final_feature_df)
    
    mlr_pred = mlr_model.predict(scaled_feature_vector)[0]
    ridge_pred = ridge_model.predict(scaled_feature_vector)[0]
    rf_pred = rf_model.predict(scaled_feature_vector)[0]
    xgb_pred = xgb_model.predict(scaled_feature_vector)[0]
    
    # ARIMA Model
    # Update the loaded ARIMA model with the full 'close' history and predict
    #arima_model.update(df['close'].dropna())
    #arima_pred = arima_model.predict(n_periods=1)[0]
    
    predictions = {
        'Linear Regression': mlr_pred,
        'Ridge Regression': ridge_pred,
        'Random Forest': rf_pred,
        'XGBoost': xgb_pred,
    #    'Auto-ARIMA': arima_pred
    }
    
    return predictions

# ==============================================================================
# 3. SCRIPT EXECUTION BLOCK
# ==============================================================================
# if __name__ == '__main__':
#     # --- DEFINE PATHS FOR PREDICTION ---

#     # Path to the folder where the trained models are saved
#     MODELS_DIRECTORY = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\models\20251103_224742'
    
#     # Paths to your NEW data files for prediction
#     QUANT_PATH = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\test_data\quant_bitcoin_20251102_2359.csv'
#     GOOGLE_PATH = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\test_data\google_news_crypto_sentiment_20251102_2355_480_hours_TEST.csv'
#     INTEREST_PATH = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\test_data\interest_rates_test_20251103_0000.csv'
    
#     # --- Run the Prediction Pipeline ---
#     df = preprocess_data(QUANT_PATH, GOOGLE_PATH, INTEREST_PATH)
#     #what we are trying to predict, doing -2 since -1 is today and the day just started.
#     true_close_value = df.iloc[-2]['close']
    
#     #historical data, not including what we are trying to predict
#     df = df.iloc[:-2].copy() 
#     all_predictions = predict_next_day(df, MODELS_DIRECTORY)

#     # --- Print the Results ---
#     if all_predictions:
#         print("\n==============================================")
#         print("Forecast for the Next Day:")
#         for model_name, pred_price in all_predictions.items():
#             print(f"  - {model_name}: {pred_price:.2f}")
#         print("==============================================")
#         print(f"True Next Day: {true_close_value}")
#         for model_name, pred_price in all_predictions.items():
#             print(f"  - {model_name} MAE: {abs(true_close_value-pred_price):.2f}")
#         print("==============================================")