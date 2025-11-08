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
    df_google['published'] = pd.to_datetime(df_google['published'], format="%a, %d %b %Y %H:%M:%S %Z", utc=True, errors='coerce')
    df_google.dropna(subset=['published'], inplace=True)
    df_google['merge_date'] = df_google['published'].dt.tz_convert(None).dt.normalize()
    
    # Aggregate to daily sentiment
    numeric_cols = df_google.select_dtypes(include=np.number).columns.tolist()
    df_google_agg = df_google.groupby('merge_date')[numeric_cols].mean().reset_index()

    # Create the 'day-ahead' key. Sentiment from Jan 1st will be applied to data from Jan 2nd.
    df_google_agg['merge_date'] = df_google_agg['merge_date'] + pd.DateOffset(days=1) # type: ignore
    
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
def train_and_evaluate(df, save_artifacts=False):
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
    # Drop the last row because it has no target. Also drop NaNs from feature engineering.
    # --- 3. Clean Data ---
    # Only drop rows where there is an NaN in any of the *required* training columns (features + target).
    # This is much safer than df.dropna() which drops based on ALL columns.
    data_clean = df

    # 🛑 Check the result after cleaning
    if data_clean.empty:
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
    selected_features = ['open', 'high', 'low', 'close', 'volume', 'weighted_sentiment', 'lag_1', 'lag_10', 
                         'rolling_mean_5', 'rolling_mean_10', 'rolling_std_10', 'volatility_7', 'momentum_5', 'high_low_spread', 'momentum_x_volume', 'rsi_sq']
    X_train = X_train[selected_features]   
    X_test = X_test[selected_features]
    features = selected_features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ########################
    
    print("--- Training and Evaluating Models ---")
    results = {}
    
    
    # Train ML Models
    mlr_model = LinearRegression().fit(X_train_scaled, y_train)
    ridge_model = Ridge(alpha=0.5, random_state=42).fit(X_train_scaled, y_train)
    rf_model = RandomForestRegressor(n_estimators=600, random_state=42).fit(X_train_scaled, y_train)
    xgb_model = xgb.XGBRegressor(colsample_bytree=1.0, learning_rate=0.03, max_depth= 7,n_estimators=1200, subsample = 0.9,
                                 objective='reg:squarederror', random_state=42).fit(X_train_scaled, y_train)
    
    # Train Auto-ARIMA on the training data for a fair comparison
    #arima_model = ARIMA(y_train, order=(1,1,3))
    #arima_model_fit = arima_model.fit()
    
    # Evaluate Models
    mlr_preds = mlr_model.predict(X_test_scaled)
    ridge_preds = ridge_model.predict(X_test_scaled)
    rf_preds = rf_model.predict(X_test_scaled)
    xgb_preds = xgb_model.predict(X_test_scaled)
    #arima_preds = arima_model_fit.forecast(steps=len(y_test))
    
    results['Linear Regression'] = {'MAE': mean_absolute_error(y_test, mlr_preds)}
    results['Ridge Regression (0.5)'] = {'MAE': mean_absolute_error(y_test, ridge_preds)}
    results['Random Forest'] = {'MAE': mean_absolute_error(y_test, rf_preds)}
    results['XGBoost'] = {'MAE': mean_absolute_error(y_test, xgb_preds)}
    
    #dropped ARIMA model. Models with more context are outperforming ARIMA. 
    #results[f'ARIMA (1,1,3)'] = {'MAE': mean_absolute_error(y_test, arima_preds)}
    
    results_df = pd.DataFrame(results).T
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
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Artifacts will be saved in: {save_dir}")

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

# def train(QUANT_PATH, GOOGLE_PATH, INTEREST_PATH) -> None:
#     # 1. Preprocess the raw data
#     preprocessed_df = preprocess_data(QUANT_PATH, GOOGLE_PATH, INTEREST_PATH)
#     # 2. Engineer features
#     featured_df = feature_engineering(preprocessed_df)
#     # 3. Train models and save artifacts if desired
#     # Set save_artifacts to True to save the models for your prediction script
#     train_and_evaluate(featured_df, save_artifacts=True)
#     print("Models trained successfully.")
