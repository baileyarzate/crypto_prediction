import sys
import os
import importlib
from main_scripts import train
importlib.reload(train)
sys.path.append(os.getcwd())
# Import the specific function 'train_and_evaluate' from the 'train' module
from main_scripts.train import preprocess_data, train_and_evaluate
from main_scripts.test import predict_next_day
from helpers.feature_engineering import feature_engineering
from helpers.data_ingestation import get_interest_data, get_quant_data, extract_google_sentiment

def run_training_pipeline(quant_path, google_path, interest_path, save = True):
    """A wrapper function to run the full training pipeline."""
    df_preprocessed = preprocess_data(quant_path, google_path, interest_path)
    df_featured = feature_engineering(df_preprocessed)
    train_and_evaluate(df_featured, save_artifacts=save)
    
def run_prediction_pipeline(quant_path, google_path, interest_path, models_dir):
    """
    Full pipeline to preprocess new data, load artifacts, and make a prediction.
    """
    # 1. Preprocess Data
    df_preprocessed = preprocess_data(quant_path, google_path, interest_path)
    print(df_preprocessed[['open','close','high','low','weighted_sentiment']])

    # Assuming the last row is "today" and has an incomplete 'close'
    
    #inference when true value exists
    #true_close_value = df_preprocessed.iloc[-2]['close'] # The value we want to compare against
    #historical_df = df_preprocessed.iloc[:-2] # Use all data UP TO and not including the day we are predicting
    
    #forecasting
    historical_df = df_preprocessed

    # 3. Make predictions
    # (This is the logic from your old predict_next_day function)
    print("--- Starting Prediction ---")
    all_predictions = predict_next_day(historical_df, models_dir)
    # --- Print the Results ---
    if all_predictions:
        print("\n==============================================")
        print("Forecast for the Next Day:")
        for model_name, pred_price in all_predictions.items():
            print(f"  - {model_name}: {pred_price:.2f}")
        print("==============================================")
        try:
            print(f"True Next Day: {true_close_value}")
            for model_name, pred_price in all_predictions.items():
                print(f"  - {model_name} Error: {(true_close_value-pred_price):.2f}")
            print("==============================================")
        except:
            print("Forecasting, no inference.")
        
if __name__ == '__main__':
    train_models = False
    save_models = False
    run_inference = True
    forecast = False
    
    if train_models:
        QUANT_PATH = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\data\1761320833.787191_year_daily.csv'
        GOOGLE_PATH = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\data\google_news_crypto_sentiment_20251030_2231_8600_hours.csv'
        INTEREST_PATH = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\data\interest_rates.csv'
        run_training_pipeline(QUANT_PATH, GOOGLE_PATH, INTEREST_PATH, save = save_models)
    
    if run_inference:
        try:# Path to the folder where the trained models are saved
            MODELS_DIRECTORY = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\models\20251104_002818'
            # Paths to your NEW data files for prediction
            QUANT_PATH = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\test_data\quant_bitcoin_test_20251104_0048.csv'#r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\test_data\quant_bitcoin_20251102_2359.csv'
            GOOGLE_PATH = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\test_data\google_news_sentiment_test_20251104_0057_hours_400.csv'#r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\test_data\google_news_crypto_sentiment_20251102_2355_480_hours_TEST.csv'
            INTEREST_PATH = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\test_data\interest_rates_test_20251104_0057.csv'#r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\test_data\interest_rates_test_20251103_0000.csv'
            run_prediction_pipeline(QUANT_PATH, GOOGLE_PATH, INTEREST_PATH, MODELS_DIRECTORY)
        except Exception as e:
            print(f"Exception Caught: {e}")
    
    if forecast:
        MODELS_DIRECTORY = r'C:\Users\1617290819C\OneDrive - United States Air Force\Documents\PersonalProjects\BitcoinPred\models\20251104_002818'
        # 1. Ingest new data and get the new file paths
        print("\nStep 1: Ingesting fresh data...")
        new_quant_path = get_quant_data()
        new_google_path = extract_google_sentiment()
        new_interest_path = get_interest_data()

        # 2. Run inference using all available historical data to predict tomorrow
        print("\nStep 3: Running inference...")
        # We pass the full new dataframe. The inference function will use the last row's features.
        run_prediction_pipeline(new_quant_path, new_google_path, new_interest_path, MODELS_DIRECTORY)