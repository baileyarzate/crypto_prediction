import sys
import os
import importlib
import argparse
from main_scripts import train
importlib.reload(train)
sys.path.append(os.getcwd())
from main_scripts.train import preprocess_data, train_and_evaluate
from main_scripts.test import predict_next_day
from helpers.feature_engineering import feature_engineering
from helpers.data_ingestation import (
    get_interest_data,
    get_quant_data,
    extract_google_sentiment,
)


def run_training_pipeline(quant_path, google_path, interest_path, save=True):
    df_preprocessed = preprocess_data(quant_path, google_path, interest_path)
    df_featured = feature_engineering(df_preprocessed)
    train_and_evaluate(df_featured, save_artifacts=save)


def run_prediction_pipeline(quant_path, google_path, interest_path, models_dir):
    df_preprocessed = preprocess_data(quant_path, google_path, interest_path)
    print(df_preprocessed[['datetime_utc','open', 'close', 'high', 'low', 'weighted_sentiment']])

    historical_df = df_preprocessed

    print("--- Starting Prediction ---")
    all_predictions = predict_next_day(historical_df, models_dir)
    if all_predictions:
        print("\n==============================================")
        print("Forecast for the Next Day:")
        for model_name, pred_price in all_predictions.items():
            print(f"  - {model_name}: {pred_price:.2f}")
        print("==============================================")


def ingest_paths(
    save_dir=None,
    # quant
    exchange="coinbase",
    symbol="BTC/USD",
    timeframe="1d",
    lookback_days=365,
    # news
    hours=400,
    batch_size=32,
    model="kk08/CryptoBERT",
    # interest
    start_date=None,
    ir_lookback_days=365,
):
    print("\nStep 1: Ingesting fresh data...")
    quant_path = get_quant_data(
        save=True,
        save_dir=save_dir,
        exchange_name=exchange,
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
    )
    google_path = extract_google_sentiment(
        hours=hours,
        save=True,
        save_dir=save_dir,
        batch_size=batch_size,
        model_name=model,
    )
    interest_path = get_interest_data(
        save=True,
        save_dir=save_dir,
        start_date=start_date,
        lookback_days=ir_lookback_days,
    )
    return quant_path, google_path, interest_path


def build_parser():
    parser = argparse.ArgumentParser(description="BitcoinPred pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # Common ingestion args helper
    def add_ingest_args(p):
        p.add_argument("--save-dir", default=None)
        # quant
        p.add_argument("--exchange", default="coinbase")
        p.add_argument("--symbol", default="BTC/USD")
        p.add_argument("--timeframe", default="1d")
        p.add_argument("--lookback-days", type=int, default=365)
        # news
        p.add_argument("--hours", type=int, default=400)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--model", default="kk08/CryptoBERT")
        # interest
        p.add_argument("--start-date", default=None)
        p.add_argument("--ir-lookback-days", type=int, default=365)

    # train
    p_train = sub.add_parser("train", help="Train models")
    group_t = p_train.add_mutually_exclusive_group(required=False)
    group_t.add_argument("--ingest", action="store_true", help="Ingest fresh data")
    group_t.add_argument("--paths", nargs=3, metavar=("QUANT", "GOOGLE", "INTEREST"))
    p_train.add_argument("--save-models", action="store_true")
    add_ingest_args(p_train)

    # predict
    p_pred = sub.add_parser("predict", help="Predict next day")
    p_pred.add_argument("--models-dir", required=True)
    group_p = p_pred.add_mutually_exclusive_group(required=False)
    group_p.add_argument("--ingest", action="store_true")
    group_p.add_argument("--paths", nargs=3, metavar=("QUANT", "GOOGLE", "INTEREST"))
    add_ingest_args(p_pred)

    # forecast (always ingest)
    p_fc = sub.add_parser("forecast", help="Ingest and predict")
    p_fc.add_argument("--models-dir", required=True)
    add_ingest_args(p_fc)

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        if args.ingest:
            q, g, i = ingest_paths(
                save_dir=args.save_dir,
                exchange=args.exchange,
                symbol=args.symbol,
                timeframe=args.timeframe,
                lookback_days=args.lookback_days,
                hours=args.hours,
                batch_size=args.batch_size,
                model=args.model,
                start_date=args.start_date,
                ir_lookback_days=args.ir_lookback_days,
            )
        elif args.paths:
            q, g, i = args.paths
        else:
            parser.error("Provide --ingest or --paths QUANT GOOGLE INTEREST")
        run_training_pipeline(q, g, i, save=args.save_models)

    elif args.cmd == "predict":
        if args.ingest:
            q, g, i = ingest_paths(
                save_dir=args.save_dir,
                exchange=args.exchange,
                symbol=args.symbol,
                timeframe=args.timeframe,
                lookback_days=args.lookback_days,
                hours=args.hours,
                batch_size=args.batch_size,
                model=args.model,
                start_date=args.start_date,
                ir_lookback_days=args.ir_lookback_days,
            )
        elif args.paths:
            q, g, i = args.paths
        else:
            parser.error("Provide --ingest or --paths QUANT GOOGLE INTEREST")
        run_prediction_pipeline(q, g, i, args.models_dir)

    elif args.cmd == "forecast":
        q, g, i = ingest_paths(
            save_dir=args.save_dir,
            exchange=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            hours=args.hours,
            batch_size=args.batch_size,
            model=args.model,
            start_date=args.start_date,
            ir_lookback_days=args.ir_lookback_days,
        )
        run_prediction_pipeline(q, g, i, args.models_dir)
    else:
        # Fallback to previous behavior if no subcommand provided
        train_models = False
        save_models = False
        run_inference = True
        forecast = False

        if train_models:
            QUANT_PATH = r''
            GOOGLE_PATH = r''
            INTEREST_PATH = r''
            run_training_pipeline(QUANT_PATH, GOOGLE_PATH, INTEREST_PATH, save=save_models)

        if run_inference:
            try:
                MODELS_DIRECTORY = r'C:\Users\baile\Documents\Artificial Intelligence\BitcoinPred\models\20251104_021130'
                QUANT_PATH = r'C:\Users\baile\Documents\Artificial Intelligence\BitcoinPred\test_data\quant_bitcoin_test_20251104_0048.csv'
                GOOGLE_PATH = r'C:\Users\baile\Documents\Artificial Intelligence\BitcoinPred\test_data\google_news_sentiment_test_20251104_0057_hours_400.csv'
                INTEREST_PATH = r'C:\Users\baile\Documents\Artificial Intelligence\BitcoinPred\test_data\interest_rates_test_20251104_0057.csv'
                run_prediction_pipeline(QUANT_PATH, GOOGLE_PATH, INTEREST_PATH, MODELS_DIRECTORY)
            except Exception as e:
                print(f"Exception Caught: {e}")

        if forecast:
            MODELS_DIRECTORY = r''
            print("\nStep 1: Ingesting fresh data...")
            new_quant_path = get_quant_data()
            new_google_path = extract_google_sentiment()
            new_interest_path = get_interest_data()
            print("\nStep 3: Running inference...")
            run_prediction_pipeline(new_quant_path, new_google_path, new_interest_path, MODELS_DIRECTORY)
