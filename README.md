# Bitcoin Prediction

Predicts the next-day Bitcoin close using classical ML models and engineered features from:
- Quant OHLCV prices (ccxt)
- Google News sentiment (optionally via a local CryptoBERT sentiment pipeline)
- US Treasury average interest rates (Fiscal Data API)

Includes end-to-end ingestion, training, artifact saving, and prediction via a simple CLI.

---

## Overview

**Ingestion**
- Prices from the selected exchange/symbol/timeframe.
- Google News sentiment aggregated day-by-day with optional local sentiment scoring.
- US Treasury interest rates pulled year-by-year.

**Preprocessing**
- Aligns interest rates from month M−1 to month M.
- Aligns daily news sentiment from day D−1 to day D (day-ahead merge).

**Feature Engineering**
- Lagged prices, rolling means/std, RSI, volatility, momentum, and simple interactions.

**Models**
- Linear Regression, Ridge, Random Forest, XGBoost
- StandardScaler and curated feature list are saved with each run.

**Prediction**
- Loads most recent historical data, recomputes features, scales, and produces next-day close forecasts from saved models.

---

## Environment Setup

**Conda (recommended)**
```bash
conda env create -f environment.yml
conda activate crypto_env
```
```yaml

**Notes**
- Google News sentiment: attempts to load Hugging Face model `kk08/CryptoBERT` locally; falls back with zero sentiment if unavailable.
- ccxt uses public OHLCV endpoints; no API keys required for basic use.
```
---
## Repository Structure
main.py CLI entrypoint for training, prediction, and forecast
main_scripts/train.py Preprocessing, feature selection, training, evaluation, artifact saving
main_scripts/test.py Loads artifacts, recomputes features, predicts next-day close
helpers/data_ingestation.py Ingests price, interest rate, and news sentiment data
helpers/feature_engineering.py RSI, lags, rolling stats, volatility, momentum features
helpers/llm_support.py LLM prompt builder and citation helper
helpers/queries.py Predefined news/social queries
analysis/true_vs_predicted.ipynb Compares predictions vs actual closes
next_day_forecast_llm.ipynb Notebook driving prediction pipeline and LLM prompt
models/ Timestamped folders with saved artifacts
data/ Timestamped subfolders created by ingestion
standalone_training/ Self-contained scripts for reproducible training/evaluation

---

## Quick Usage

**Train with fresh data and save models**
```bash
python main.py train --ingest --save-models
# Optional flags: `--exchange`, `--symbol`, `--timeframe`, `--lookback-days`, `--hours`, `--batch-size`, `--model`, `--max-results-per-query`, `--ir-lookback-days`, `--save-dir`
```

**Predict using existing models and fresh ingestion**
```bash
python main.py predict --models-dir models/20251109_210801 --ingest
```

**Predict using curated file paths**
```bash
python main.py predict --models-dir models/20251109_210801 --paths PATH_TO_QUANT.csv PATH_TO_GOOGLE.csv PATH_TO_INTEREST.csv
```

**End-to-end forecast**

---

## Standalone Ingestion

**Quant OHLCV**
```bash
python helpers/data_ingestation.py quant --exchange coinbase --symbol "BTC/USD" --timeframe 1d --lookback-days 1095 --save-dir data
```

**Interest rates**
```bash
python helpers/data_ingestation.py interest --lookback-days 1095 --save-dir data
```

**Google News sentiment**
```bash
python helpers/data_ingestation.py news --hours 26280 --batch-size 32 --model kk08/CryptoBERT --max-results-per-query 1 --save-dir data
```

---

## Standalone Training
**Purpose**
- Run small, self-contained training/evaluation scripts decoupled from CLI.
- Mirrors main pipeline defaults: ingestion → preprocessing → features → training → artifacts.

**Location**
- `standalone_training/`
- Scripts print usage with `--help` and save artifacts under `models/`.

**Typical usage**
```bash
python standalone_training/train.py --help
python standalone_training/train.py --ingest --save-models
python standalone_training/train.py --paths PATH_TO_QUANT.csv PATH_TO_GOOGLE.csv PATH_TO_INTEREST.csv --save-models
```

**Notes**
- Run from repo root to resolve paths/imports correctly.
- Outputs align with main CLI: timestamped subfolders under `models/` and `data/` unless overridden.

---

## How It Works

**Preprocessing (`main_scripts/train.py`)**
- `merge_date` normalized from `datetime_utc`
- Interest rates pivoted by `security_desc` and shifted 1 month forward
- News sentiment aggregated by day, shifted 1 day forward

**Features (`helpers/feature_engineering.py`)**
- RSI(14), lags (1/5/10), rolling mean/std (5/10), volatility(7), momentum(5)
- High-low spread, momentum×volume, rsi²

**Training**
- Time-ordered train/test split
- Scaler fit on train
- Models trained and evaluated (MAE)
- Artifacts saved under `models/{YYYYMMDD_HHMMSS}/`

**Prediction**
- Reloads models, scaler, feature list
- Recomputes features on latest data
- Returns dict of model predictions

---

## Data and Outputs

**Ingestion output**
```bash
data/{YYYYMMDD_HHMMSS}/quant/quant_bitcoin_test_.csv
data/{YYYYMMDD_HHMMSS}/sentiment/google_news_sentiment_.csv
data/{YYYYMMDD_HHMMSS}/interest/interest_rates_test_*.csv
```

**Models**
```bash
models/{YYYYMMDD_HHMMSS}/[mlr|ridge|rf|xgb]_model.joblib
scaler.joblib
feature_list.joblib
```
---

## Notebooks

- `next_day_forecast_llm.ipynb`: runs prediction pipeline, prints forecasts, builds LLM prompt
- `analysis/true_vs_predicted.ipynb`: side-by-side predictions vs actual closes

---

## Configuration

- `SAVE_DIR` env var overrides default data save location (otherwise under `data/`)
- Timestamps handled in UTC; aligns UTC close with previous-day 16:00 PST

---

## Troubleshooting

**Sentiment model unavailable**
- Pipeline sets `weighted_sentiment=0.0` and logs a warning

**Notebook tqdm warnings**
- Install/enable `ipywidgets` to remove

**Feature engineering guard**
- Requires at least 15 rows of history; otherwise prediction aborts

**Known issues**
- Malformed f-string in `helpers/feature_engineering.py` error print
- Some console output has garbled characters from copy/paste

---

## Optional LLM Support

**helpers/llm_support.py**
- `get_prompt(predictions, yesterdays_close)`: structured forecasting prompt for LLM
- `add_citations(response)`: attach citation links if metadata available
