# Training Data Guide

This folder provides a ready-to-train, pre-merged dataset and a notebook that trains models directly from that file. It is intended for users who want to bypass the ingestion/merging pipeline in `main.py` and train quickly from a single CSV.

## What’s Here

- `bitcoin_sentiment_12012022_11082025.csv`
  - Daily BTC OHLCV plus aligned features (interest-rate columns and `weighted_sentiment`).
  - Columns include: `timestamp, open, high, low, close, volume, datetime_utc, merge_date, ... interest_* columns ..., weighted_sentiment, sentiment_missing`.
- `train_with_training_data.ipynb`
  - A self-contained notebook that loads this CSV, engineers features, trains the suite of ML models, and can save artifacts.

## When To Use This

- You want to train from a single, already-aligned dataset without running the ingestion steps.
- You have your own similarly structured CSV and want to train models quickly.

If you want the full ingestion → preprocess → feature engineering → training flow, use the CLI in `main.py` instead.

## Quick Start (Notebook)

1. Open `training_data/train_with_training_data.ipynb`.
2. Set the CSV path to `training_data/bitcoin_sentiment_12012022_11082025.csv` (or your own file).
3. Run all cells to:
   - Load data
   - Engineer features (mirrors the features used in `helpers/feature_engineering.py`)
   - Train models (Linear Regression, Ridge, Random Forest, XGBoost)
   - Optionally save artifacts (`*.joblib`) to a timestamped folder under `models/`

## Bring Your Own Dataset

Your CSV should provide at least:
- Price series: `open, high, low, close, volume`
- Time keys: `datetime_utc` (UTC timestamp) and/or `merge_date` (daily date)
- Sentiment: `weighted_sentiment` (float; set 0.0 if unavailable) and optional `sentiment_missing` flag
- Interest rates: one or more numeric columns (e.g., `treasury_bills`, `treasury_notes`, etc.). If not available, you can drop these from the feature list in the notebook.

Notes:
- The feature engineering step requires a minimum history (~15 rows) to compute rolling/lagged features.
- If you remove columns, also update the selected feature list before training.

## Outputs and Using Models

- After training with the notebook, artifacts are saved to `models/{YYYYMMDD_HHMMSS}/` as:
  - `mlr_model.joblib`, `ridge_model.joblib`, `rf_model.joblib`, `xgb_model.joblib`, `scaler.joblib`, and `feature_list.joblib`.
- You can use these models with the main prediction pipeline:
  - `python main.py predict --models-dir models/{YYYYMMDD_HHMMSS} --paths <QUANT.csv> <GOOGLE.csv> <INTEREST.csv>`
  - Or use `--ingest` to fetch fresh data before predicting.

## Important Compatibility Notes

- The main `preprocess_data` function in `main_scripts/train.py` expects three separate files (quant, sentiment, interest) or freshly ingested data. The training notebook here bypasses that by starting from a single, already-merged CSV.
- If you want to train via `main.py` using a single merged CSV, you will need to modify or replace the training entry point to load your merged file directly (the current CLI merges three sources by design).

## Troubleshooting

- Not enough rows for features:
  - Ensure your CSV has at least ~15 consecutive daily rows.
- Missing columns:
  - Remove/replace those from the feature list in the notebook and re-run.
- Artifacts not saving:
  - Confirm the notebook uses a valid output directory under `models/` and you have write permissions.
