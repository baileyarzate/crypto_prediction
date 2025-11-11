Title: BitcoinPred — Next‑Day BTC Close Prediction
One‑paragraph overview of the pipeline and goals
Quickstart with conda/pip setup
CLI usage for train/predict/forecast and standalone ingestion
How it works (ingestion → preprocessing → features → training → artifacts → prediction)
Repo structure summary
Notebooks and analysis notes
Troubleshooting and known caveats
Optional LLM prompt integration
Data locations and outputs
Here’s the content:

BitcoinPred

Predicts the next-day Bitcoin close using classical ML models and engineered features from:
Quant OHLCV prices (ccxt)
Google News sentiment (optionally via a local CryptoBERT sentiment pipeline)
US Treasury average interest rates (Fiscal Data API)
Includes end-to-end ingestion, training, artifact saving, and prediction via a simple CLI.
Overview

Ingestion:
Prices from the selected exchange/symbol/timeframe.
Google News sentiment, day-by-day windows with optional local sentiment scoring.
US Treasury interest rates pulled year-by-year.
Preprocessing:
Aligns interest rates from month M−1 to month M.
Aligns daily news sentiment from day D−1 to day D (day-ahead merge).
Feature engineering:
Lagged prices, rolling means/std, RSI, volatility, momentum, and simple interactions.
Models:
Linear Regression, Ridge, Random Forest, XGBoost
StandardScaler and a curated feature list are saved with each run.
Prediction:
Loads most recent historical data, re-computes features, scales, and produces next-day close forecasts from saved models.
Environment Setup

Conda (recommended)
Create environment: conda env create -f environment.yml
Activate: conda activate crypto_env
Pip (alternative)
Within a Python 3.11 venv: pip install -r requirements.txt
Notes

For Google News sentiment, the code attempts to load Hugging Face model “kk08/CryptoBERT” with local_files_only. If not present, it falls back gracefully with zero sentiment and logs a warning.
ccxt uses public endpoints for OHLCV by default; no API keys are required for basic use.
Repository Structure

main.py: CLI entrypoint for training, prediction, and forecast (ingest + predict).
main_scripts/train.py: Preprocessing, feature selection, training, evaluation, artifact saving.
main_scripts/test.py: Loads artifacts, re-computes features, and predicts next-day close.
helpers/data_ingestation.py: Ingests price data (ccxt), interest rates (Treasury API), Google News + sentiment.
helpers/feature_engineering.py: RSI, lags, rolling stats, volatility, momentum features.
helpers/llm_support.py: Prompt builder and citation helper for LLM post-analysis.
helpers/queries.py: Predefined queries for news sources and social channels.
analysis/true_vs_predicted.ipynb: Compares model predictions to actual closes.
next_day_forecast_llm.ipynb: Notebook driving the prediction pipeline and building an LLM prompt.
models/: Timestamped folders with saved artifacts (mlr_model.joblib, ridge_model.joblib, rf_model.joblib, xgb_model.joblib, scaler.joblib, feature_list.joblib).
data/: Timestamped subfolders created by ingestion (quant/, sentiment/, interest/).
Quick Usage

Train with fresh data and save models
python main.py train --ingest --save-models
Useful flags: --exchange coinbase --symbol "BTC/USD" --timeframe 1d --lookback-days 1095 --hours 26280 --batch-size 32 --model kk08/CryptoBERT --max-results-per-query 1 --ir-lookback-days 1095 --save-dir data
Predict using existing models and fresh ingestion
python main.py predict --models-dir models/20251109_210801 --ingest
Predict using curated file paths (no ingestion)
python main.py predict --models-dir models/20251109_210801 --paths PATH_TO_QUANT.csv PATH_TO_GOOGLE.csv PATH_TO_INTEREST.csv
End-to-end forecast (always ingests then predicts)
python main.py forecast --models-dir models/20251109_210801
Standalone Ingestion

Quant OHLCV:
python helpers/data_ingestation.py quant --exchange coinbase --symbol "BTC/USD" --timeframe 1d --lookback-days 1095 --save-dir data
Interest rates (Treasury):
python helpers/data_ingestation.py interest --lookback-days 1095 --save-dir data
Google News sentiment:
python helpers/data_ingestation.py news --hours 26280 --batch-size 32 --model kk08/CryptoBERT --max-results-per-query 1 --save-dir data
How It Works

Preprocessing (main_scripts/train.py:80+)
merge_date normalized from datetime_utc
Interest pivoted by security_desc and shifted one month forward; merged on interest_month_key
News sentiment aggregated by day and shifted one day forward; merged on merge_date
Features (helpers/feature_engineering.py)
RSI(14), lags (1/5/10), rolling means/std (5/10), volatility(7), momentum(5), high-low spread, momentum×volume, rsi^2
Training (main_scripts/train.py)
Train/test split (time-ordered), scaler fit on train, models trained and evaluated (MAE)
Artifacts saved under models/{YYYYMMDD_HHMMSS}/
Prediction (main_scripts/test.py)
Reloads models, scaler, and feature list; recomputes features on latest data; returns a dict of model predictions.
Data and Outputs

Ingestion output directory:
data/{YYYYMMDD_HHMMSS}/quant/quant_bitcoin_test_*.csv
data/{YYYYMMDD_HHMMSS}/sentiment/google_news_sentiment_*.csv
data/{YYYYMMDD_HHMMSS}/interest/interest_rates_test_*.csv
Models:
models/{YYYYMMDD_HHMMSS}/[mlr|ridge|rf|xgb]_model.joblib, scaler.joblib, feature_list.joblib
Notebooks

next_day_forecast_llm.ipynb: Drives run_prediction_pipeline(...), prints model forecasts, and builds an LLM prompt via helpers/llm_support.get_prompt.
analysis/true_vs_predicted.ipynb with analysis/true_vs_predicted_data.csv: Displays side-by-side predictions vs. actual closes (includes Gemini estimates from notes when present).
Configuration

SAVE_DIR env var to override the default data save location (otherwise under data/).
Timezone: timestamps are handled in UTC; the project notes align UTC close with previous-day 16:00 PST.
Troubleshooting

Sentiment model unavailable
You’ll see a warning and the pipeline sets weighted_sentiment=0.0. To enable local inference, download “kk08/CryptoBERT” and ensure Hugging Face caches are available offline.
Notebook tqdm warnings
Install/enable ipywidgets to remove tqdm warnings in Jupyter.
Feature engineering guard
Prediction requires at least 15 rows of history; otherwise it aborts.
Known issues to tidy
helpers/feature_engineering.py has a malformed f-string in an error print; fix printing or remove it.
Some console output contains garbled characters from copy/paste; harmless but can be cleaned up.
Optional LLM Support

helpers/llm_support.py
get_prompt(predictions, yesterdays_close) constructs a structured forecasting prompt for use with an LLM.
add_citations(response) can attach citation links for grounded responses (e.g., Gemini), if metadata is available.
