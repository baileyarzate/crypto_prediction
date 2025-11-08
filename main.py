import sys
import os
from datetime import datetime
import pytz
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


def run_training_pipeline(
    quant_path: str | None = None,
    google_path: str | None = None,
    interest_path: str | None = None,
    save: bool = True,
    # If ingest=True or any path is None, we'll call ingest_paths with options below
    ingest: bool = False,
    # Ingestion options (mirrors ingest_paths args)
    save_dir: str | None = None,
    exchange: str = "coinbase",
    symbol: str = "BTC/USD",
    timeframe: str = "1d",
    lookback_days: int = 1095,
    hours: int = 26280,
    batch_size: int = 32,
    model: str = "kk08/CryptoBERT",
    max_results_per_query: int | None = None,
    start_date: str | None = None,
    ir_lookback_days: int = 1095,
):
    """
    Train models using either curated file paths or freshly ingested data.

    - If all three paths are provided and ingest is False, uses curated data.
    - Otherwise (ingest=True or any path is None), ingests new data first.
    """
    if ingest or not (quant_path and google_path and interest_path):
        quant_path, google_path, interest_path = ingest_paths(
            save_dir=save_dir,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            hours=hours,
            batch_size=batch_size,
            model=model,
            max_results_per_query=max_results_per_query,
            start_date=start_date,
            ir_lookback_days=ir_lookback_days,
        )

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
    lookback_days=1095,
    # news
    hours=26280,
    batch_size=32,
    model="kk08/CryptoBERT",
    max_results_per_query=None,
    # interest
    start_date=None,
    ir_lookback_days=1095,
):
    print("\nStep 1: Ingesting fresh data...")

    # Prepare a runtime-scoped directory under test_data (or provided save_dir)
    if save_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    else:
        base_dir = save_dir
    run_id = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    quant_dir = os.path.join(run_dir, 'quant')
    sentiment_dir = os.path.join(run_dir, 'sentiment')
    interest_dir = os.path.join(run_dir, 'interest')
    try:
        os.makedirs(quant_dir, exist_ok=True)
        os.makedirs(sentiment_dir, exist_ok=True)
        os.makedirs(interest_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning creating run directories: {e}")
    print(f"Saving this run's data under: {run_dir}")

    quant_path = get_quant_data(
        save=True,
        save_dir=quant_dir,
        exchange_name=exchange,
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
    )
    google_path = extract_google_sentiment(
        hours=hours,
        save=True,
        save_dir=sentiment_dir,
        batch_size=batch_size,
        model_name=model,
        max_results_per_query=max_results_per_query,
    )
    interest_path = get_interest_data(
        save=True,
        save_dir=interest_dir,
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
        p.add_argument("--lookback-days", type=int, default=1095)
        # news
        p.add_argument("--hours", type=int, default=26280)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--model", default="kk08/CryptoBERT")
        p.add_argument("--max-results-per-query", type=int, default=1)
        # interest
        p.add_argument("--start-date", default=None)
        p.add_argument("--ir-lookback-days", type=int, default=1095)

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
                max_results_per_query=args.max_results_per_query,
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
                max_results_per_query=args.max_results_per_query,
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
            max_results_per_query=args.max_results_per_query,
            start_date=args.start_date,
            ir_lookback_days=args.ir_lookback_days,
        )
        run_prediction_pipeline(q, g, i, args.models_dir)
    else:
        # Fallback to previous behavior if no subcommand provided
        train_models = True
        save_models = True
        run_inference = False
        forecast = False

        if train_models:
            run_training_pipeline(ingest=True,
                                  lookback_days = 1095,
                                  hours = 26280,
                                  batch_size = 32,
                                  model = "kk08/CryptoBERT",
                                  ir_lookback_days= 1095,
                                  save = save_models,
                                  max_results_per_query = 2)

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
