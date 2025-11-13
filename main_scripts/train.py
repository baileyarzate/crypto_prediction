import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
#from statsmodels.tsa.arima.model import ARIMA
import logging
# --- Setup Logging ---
# Define the log file path relative to the current script's parent directory
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'train_logs.txt')
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a'), # Append to the file
        logging.StreamHandler() # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# 1. DATA PREPROCESSING FUNCTION
import pandas as pd
import numpy as np

def preprocess_data(quant_path, google_path, interest_path):
    """
    Implements the specific temporal alignment logic:
    - Quant data is the base.
    - Interest rates from month M-1 are applied to quant data in month M.
    - Google sentiment from day D-1 is applied to quant data on day D.
    """
    print("--- Starting Data Preprocessing with Temporal Alignment ---")
    
    # --- 1. Load All Data ---
    df_quant = pd.read_csv(quant_path)
    df_google = pd.read_csv(google_path)
    df_interest = pd.read_csv(interest_path)

    # --- 2. Prepare Base DataFrame (df_quant) ---
    df_quant['merge_date'] = pd.to_datetime(df_quant['datetime_utc'], utc=True).dt.tz_convert(None).dt.normalize()
    # Create the key for the upcoming month-based interest rate merge
    df_quant['interest_month_key'] = df_quant['merge_date'].dt.strftime('%Y-%m')

    # --- 3. Prepare and Align Interest Rate Data ---
    # Pivot the interest data
    df_interest_wide = df_interest.pivot(index='record_date', columns='security_desc', values='avg_interest_rate_amt').reset_index()
    df_interest_wide.columns = [col.lower().replace(' ', '_') for col in df_interest_wide.columns]
    
    # Create the look-ahead key
    df_interest_wide['record_date'] = pd.to_datetime(df_interest_wide['record_date'])
    df_interest_wide['interest_month_key'] = (df_interest_wide['record_date'] + pd.DateOffset(months=1)).dt.strftime('%Y-%m') # type: ignore
    
    # Merge based on the month key
    df_merged = pd.merge(df_quant, df_interest_wide.drop(columns='record_date'), on='interest_month_key', how='left')

    # --- 4. Prepare and Align Google Sentiment Data ---
    # Process dates
    try:
        df_google['published'] = pd.to_datetime(df_google['published'], format="%a, %d %b %Y %H:%M:%S %Z", utc=True, errors='coerce')
        df_google.dropna(subset=['published'], inplace=True)
        df_google['merge_date'] = df_google['published'].dt.tz_convert(None).dt.normalize()
    except: pass
    
    # Aggregate to daily sentiment
    numeric_cols = df_google.select_dtypes(include=np.number).columns.tolist()
    df_google_agg = df_google.groupby('merge_date')[numeric_cols].mean().reset_index()

    # Create the 'day-ahead' key. Sentiment from Jan 1st will be applied to data from Jan 2nd.
    #df_google_agg['merge_date'] = df_google_agg['merge_date'] + pd.DateOffset(days=1) # type: ignore
    df_google_agg['merge_date'] = pd.to_datetime(df_google_agg['merge_date'], errors='coerce')
    df_google_agg['merge_date'] = df_google_agg['merge_date'] + pd.DateOffset(days=1)

    
    # Now, merge this aligned sentiment data
    df_final = pd.merge(df_merged, df_google_agg, on='merge_date', how='left')

    # Ensure sentiment column exists and handle API-limit gaps
    if 'weighted_sentiment' not in df_final.columns:
        df_final['weighted_sentiment'] = 0.0
        df_final['sentiment_missing'] = 1
    else:
        # Flag where sentiment was originally missing, then fill with zeros
        missing_mask = df_final['weighted_sentiment'].isna()
        df_final['sentiment_missing'] = missing_mask.astype(int)
        df_final['weighted_sentiment'] = df_final['weighted_sentiment'].fillna(0.0)

    # --- 5. Final Filtering and Cleaning ---
    # Before dropping, sort and ffill 
    df_final = df_final.sort_values(by='merge_date').reset_index(drop=True)
    
    # Forward-fill NaNs that might exist from the merge 
    df_final.ffill(inplace=True)
    
    # Now, drop any remaining NaNs (sentiment handled earlier)
    df_final.dropna(inplace=True)
    
    # Clean up helper columns
    df_final.drop(columns=['interest_month_key','Unnamed: 0','time_period'], inplace=True, errors='ignore')
    
    print("--- Data Preprocessing Complete ---")
    return df_final.reset_index(drop=True)

# 3. TRAINING AND EVALUATION FUNCTION
def train_and_evaluate(df, save_artifacts=False, selected_features = None):
    """
    Trains models to predict the *next day's* close price.
    """
    print("--- Preparing Data for True Forecasting ---")
    
    # --- 1. Create the Target Variable ---
    # The target 'y' is now TOMORROW's close price.
    df['target'] = df['close'].shift(-1)
    # --- 2. Define Features ---
    # The 'close' column is now a feature, not the target.
    # We must exclude the new 'target' column from the features.
    features = [col for col in df.columns if col not in ['merge_date', 'datetime_utc', 'timestamp', 'target'] and 'unnamed' not in col]
    # --- 3. Clean Data ---
    # Define the columns that MUST NOT have NaNs for training
    required_cols = [col for col in df.columns if col not in ['merge_date', 'datetime_utc', 'timestamp', 'unnamed'] and 'unnamed' not in col]
    required_cols.append('target') # Ensure the new target is included

    # Drop rows based on the required columns
    data_clean = df.dropna(subset=required_cols).reset_index(drop=True)

    # Check the result after cleaning
    if data_clean.empty:
        logger.error("The dataset is empty after dropping NaNs.")
        raise ValueError("The dataset is empty after dropping NaNs. Check your feature engineering steps for excessive NaN creation!")
    
    X = data_clean[features]
    y = data_clean['target'] # Use the new 'target' column

    # --- 4. Split and Train ---
    # The rest of your training logic is the same...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    #######################Find best features
    # from sklearn.feature_selection import RFE

    # # ... (after creating X_train, y_train, etc.)

    # print("\n--- Performing RFE using RandomForest as the estimator ---")

    # # Use a trained RandomForest model to judge feature importance.
    # # This is much more stable than Linear Regression in the presence of correlated features.
    # estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # # Let's try selecting the top 15 features this time.
    # # It's better to start by removing only a few features.
    # selector = RFE(estimator, n_features_to_select=16, step=1, verbose=1)
    # selector = selector.fit(X_train, y_train) # Use non-scaled data for RFE

    # # Get the names of the most important features as chosen by the RandomForest
    # selected_features = X_train.columns[selector.support_]
    # print(f"\nTop 16 features selected by RFE: {selected_features.tolist()}")

    # # --- Use ONLY the selected features for the rest of the pipeline ---
    # X_train_rfe = X_train[selected_features]
    # X_test_rfe = X_test[selected_features]

    # # Scale the RFE-selected data
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train_rfe)
    # X_test_scaled = scaler.transform(X_test_rfe)
    #######################uncomment block if needed again
    logger.info("=" * 60)
    if selected_features is None: 
        selected_features = ['open', 'high', 'low', 'close', 'volume', 'weighted_sentiment', 'lag_1', 'lag_10', 
                         'rolling_mean_5', 'rolling_mean_10', 'rolling_std_10', 'volatility_7', 'momentum_5', 'high_low_spread', 'momentum_x_volume', 'rsi_sq']
    logger.info(f"\nSelected Features used for Training: {selected_features}")
    X_train = X_train[selected_features]   
    X_test = X_test[selected_features]
    features = selected_features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ########################
    
    print("--- Training and Evaluating Models ---")
    results = {}
    
    HYPERPARAMS = {
        'Linear Regression': {},
        'Ridge Regression (0.5)': {'alpha': 0.5},
        'Random Forest': {'n_estimators': 600, 'random_state': 42},
        'XGBoost': {'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 7, 'n_estimators': 1200, 'subsample': 0.9}
    }
    
    for model_name, params in HYPERPARAMS.items():
        logger.info(f"Model: {model_name}, Hyperparameters: {params}")
    # Train ML Models
    mlr_model = LinearRegression().fit(X_train_scaled, y_train)
    ridge_model = Ridge(**HYPERPARAMS['Ridge Regression (0.5)'], random_state=42).fit(X_train_scaled, y_train)
    rf_model = RandomForestRegressor(**HYPERPARAMS['Random Forest']).fit(X_train_scaled, y_train)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **HYPERPARAMS['XGBoost']).fit(X_train_scaled, y_train)
    
    # Train Auto-ARIMA on the training data for a fair comparison
    #arima_model = ARIMA(y_train, order=(1,1,3))
    #arima_model_fit = arima_model.fit()
    
    # Evaluate Models
    mlr_preds = mlr_model.predict(X_test_scaled)
    ridge_preds = ridge_model.predict(X_test_scaled)
    rf_preds = rf_model.predict(X_test_scaled)
    xgb_preds = xgb_model.predict(X_test_scaled)
    #arima_preds = arima_model_fit.forecast(steps=len(y_test))

    # prophet was giving an MAE of ~40,000
    # print("--- Training Prophet ---")
    # df_prophet = data_clean[['datetime_utc', 'target']].rename(columns={'datetime_utc': 'ds', 'target': 'y'})
    # df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], utc=True)  # convert and mark as UTC
    # df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)       # make naive
    # df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
    # from prophet import Prophet
    # prophet_model = Prophet()
    # prophet_model.fit(df_prophet[:-len(y_test)])  # Train on all but last test days
    # future = prophet_model.make_future_dataframe(periods=len(y_test), freq='D')
    # prophet_preds = prophet_model.predict(future)['yhat'].tail(len(y_test)).values
    # prophet_preds = pd.Series(prophet_preds).fillna(method='ffill').values
    

    results['Linear Regression'] = {'MAE': mean_absolute_error(y_test, mlr_preds)}
    results['Ridge Regression (0.5)'] = {'MAE': mean_absolute_error(y_test, ridge_preds)}
    results['Random Forest'] = {'MAE': mean_absolute_error(y_test, rf_preds)}
    results['XGBoost'] = {'MAE': mean_absolute_error(y_test, xgb_preds)}
    #results['Prophet'] = {'MAE': mean_absolute_error(y_test, prophet_preds)}
    
    #dropped ARIMA model. Models with more context are outperforming ARIMA. 
    #results[f'ARIMA (1,1,3)'] = {'MAE': mean_absolute_error(y_test, arima_preds)}
    
    results_df = pd.DataFrame(results).T
    logger.info(results_df.sort_values(by='MAE').to_markdown())
    print("\n--- Model Performance on Test Set ---")
    print(results_df.sort_values(by='MAE'))

    # Optional: Save artifacts
     # Optional: Save artifacts
    if save_artifacts:
        print("\n--- Saving Production Artifacts ---")

        # Create a timestamp for the run
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct the save directory path using the parent directory
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', run_timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        #print(f"Artifacts will be saved in: {save_dir}")
        logger.info(f"Artifacts will be saved in: {save_dir}")

        # Retrain final Auto-ARIMA on all available data for the best forecast
        #final_arima_model = ARIMA(y, order=(1, 1, 0)).fit()
        
        # Save all artifacts to the newly created directory
        joblib.dump(mlr_model, os.path.join(save_dir, 'mlr_model.joblib'))
        joblib.dump(ridge_model, os.path.join(save_dir, 'ridge_model.joblib'))
        joblib.dump(rf_model, os.path.join(save_dir, 'rf_model.joblib'))
        joblib.dump(xgb_model, os.path.join(save_dir, 'xgb_model.joblib'))
        joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
        joblib.dump(features, os.path.join(save_dir, 'feature_list.joblib'))
        
        print("--- Artifacts Saved Successfully ---")
    logger.info("=" * 60)