# BitcoinPred

End-to-end pipeline for forecasting Bitcoin’s next-day close. The project ingests market data, engineers technical + macro features, trains a small ensemble of classical models, and surfaces predictions in both a CLI and a Streamlit dashboard (with optional LLM commentary).

---

## Highlights

- **Multi-source ingestion**: OHLCV via ccxt, Google News sentiment with optional `kk08/CryptoBERT`, and US Treasury yield curves (Fiscal Data API).
- **Consistent feature store**: rolling stats, RSI, volatility, momentum, interest rate pivots, and day-ahead sentiment alignment.
- **Model ensemble**: Linear Regression, Ridge, Random Forest, and XGBoost with shared scaler + feature list artifacts.
- **Streamlit dashboard**: run forecasts, trigger fresh ingestion, compare model outputs, chart price history in UTC or Pacific time, and request Gemini commentary with citations.
- **Reproducible CLI + notebooks**: train/predict scripts, analysis notebooks, and timestamped `data/` + `models/` directories for every run.

---

## Getting Started

### 1. Create an environment

```bash
conda env create -f environment.yml
conda activate crypto_env
# or
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure secrets (optional)

Create `.env` in the repo root:

```
GEMINI_API_KEY=your-gemini-key
SAVE_DIR=C:\path\to\custom\data        # optional override
```

Gemini is only required if you want AI commentary inside Streamlit.

---

## Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Key workflow:
1. Pick a saved `models/` run; paths auto-populate.
2. Choose an existing `data/` snapshot or toggle **“Ingest fresh data (~10-15 min)”** to pull a new 20-day window.
3. Adjust the history slider and timezone (UTC or Pacific with DST) for the chart + preview table.
4. Run the forecast. Results persist in-session so you can tweak settings without re-running the pipeline.
5. (Optional) Toggle **AI prediction and commentary** to call Gemini; citations are appended automatically via Google Search tools.

Outputs include:
- Summary metrics (latest close, ensemble average, model spread).
- Chart with historical closes plus the Linear Regression prediction marker.
- Downloadable prediction table and feature preview with both UTC and PST timestamps.

---

## CLI Usage

The CLI mirrors the dashboard but is script-friendly.

### Train (fresh ingest + save artifacts)
```bash
python main.py train --ingest --save-models \
    --lookback-days 1095 --hours 26280 --ir-lookback-days 1095
```

### Predict with fresh ingestion
```bash
python main.py predict --models-dir models/20251112_170018 --ingest
```

### Predict from curated CSV paths
```bash
python main.py predict --models-dir models/20251112_170018 \
    --paths data/20251112_172623/quant/quant_bitcoin_test_20251112_1726.csv \
            data/20251112_172623/sentiment/google_news_sentiment_20251112_1739_days_20.csv \
            data/20251112_172623/interest/interest_rates_test_20251112_1739.csv
```

All commands accept additional switches (`--exchange`, `--symbol`, `--timeframe`, `--max-results-per-query`, etc.). Use `python main.py --help` for the full matrix.

---

## Repository Tour

| Path | Purpose |
| --- | --- |
| `main.py` | CLI entrypoint (train, predict, forecast) |
| `streamlit_app.py` | Dashboard with ingestion toggle, charting, downloads, Gemini commentary |
| `main_scripts/train.py` | Preprocess, engineer features, train/evaluate, persist artifacts |
| `main_scripts/test.py` | Feature rebuild + model inference |
| `helpers/data_ingestation.py` | Independent quant/news/interest ingestion commands |
| `helpers/feature_engineering.py` | RSI, lags, rolling stats, volatility, momentum features |
| `helpers/llm_support.py` | Prompt builder + citation helper for Gemini |
| `helpers/queries.py` | Default news/search query lists |
| `analysis/*.ipynb` | Evaluation visuals (`plots_for_readers.ipynb`, `true_vs_predicted.ipynb`, etc.) |
| `standalone_training/` | Sandbox scripts/notebooks for reproducible experiments |
| `data/{timestamp}/` | Saved CSV snapshots (quant, sentiment, interest) |
| `models/{timestamp}/` | Serialized models (`mlr`, `ridge`, `rf`, `xgb`), scaler, feature list |

---

## Data + Artifacts

Every ingest run creates:
```
data/{YYYYMMDD_HHMMSS}/quant/quant_bitcoin_test_*.csv
data/{YYYYMMDD_HHMMSS}/sentiment/google_news_sentiment_*.csv
data/{YYYYMMDD_HHMMSS}/interest/interest_rates_test_*.csv
```

Training saves to:
```
models/{YYYYMMDD_HHMMSS}/mlr_model.joblib
models/{YYYYMMDD_HHMMSS}/ridge_model.joblib
models/{YYYYMMDD_HHMMSS}/rf_model.joblib
models/{YYYYMMDD_HHMMSS}/xgb_model.joblib
models/{YYYYMMDD_HHMMSS}/scaler.joblib
models/{YYYYMMDD_HHMMSS}/feature_list.joblib
```

The Streamlit app and CLI auto-discover these directories, so keeping timestamped folders untouched preserves reproducibility.

---

## Troubleshooting

- **Sentiment model missing**: if `kk08/CryptoBERT` cannot be loaded, sentiment defaults to `0.0`; warnings are logged.
- **Feature guard**: predictions need at least 15 historical rows after preprocessing; otherwise the pipeline aborts early.
- **Gemini errors**: ensure `GEMINI_API_KEY` is set; the dashboard will gracefully disable commentary when unavailable.
- **Widget deprecations**: the app already uses the new `width` parameter (replacing `use_container_width`) to stay compatible with Streamlit >= 1.40.

---

## Notebooks

- `analysis/plots_for_readers.ipynb` – curated visuals for presentations/blog posts.
- `analysis/true_vs_predicted.ipynb` – compares model forecasts vs actual closes.
- `next_day_forecast_llm.ipynb` – scripted forecast run with LLM prompt generation.
- `standalone_training/train_with_training_data.ipynb` – reproducible training experiments outside the CLI.

---

## Roadmap Ideas

- Expand model zoo (CatBoost/LightGBM, simple LSTM baseline).
- Deploy Streamlit as a scheduled Cloud Run/Spaces app.
- Add automated backtesting metrics and alerting hooks.

Contributions, issues, and feature requests are welcome!
