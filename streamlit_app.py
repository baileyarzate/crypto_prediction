import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    import altair as alt
except Exception:  # pragma: no cover - Altair is optional
    alt = None

from google import genai
from google.genai import types

from helpers.llm_support import add_citations, get_prompt
from helpers.queries import urls
from main import run_prediction_pipeline

BASE_DIR = Path(__file__).resolve().parent
MODELS_ROOT = BASE_DIR / "models"
DATA_ROOT = BASE_DIR / "data"
DEFAULT_HISTORY_WINDOW = 30
TIMEZONE_CHOICES = {"UTC": "UTC", "Pacific (US)": "America/Los_Angeles"}


def list_runs(root: str) -> List[str]:
    path = Path(root)
    if not path.exists():
        return []
    dirs = [p for p in path.iterdir() if p.is_dir()]
    return [d.name for d in sorted(dirs, key=lambda p: p.name, reverse=True)]


def infer_data_paths(run_name: str) -> Tuple[str, str, str]:
    run_dir = DATA_ROOT / run_name
    if not run_dir.exists():
        return ("", "", "")
    return (
        _latest_csv(run_dir / "quant"),
        _latest_csv(run_dir / "sentiment"),
        _latest_csv(run_dir / "interest"),
    )


def _latest_csv(folder: Path) -> str:
    if not folder.exists():
        return ""
    files = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return str(files[-1]) if files else ""


def get_dataset_length(csv_path: str) -> int:
    if not csv_path:
        return 0
    try:
        df = pd.read_csv(csv_path, usecols=["datetime_utc"])
        return len(df)
    except Exception:
        return 0


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def build_predictions_table(predictions: Dict[str, float], last_close: float) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            {
                "Model": name,
                "Predicted Close ($)": pred,
                "Δ vs Last Close ($)": pred - last_close,
                "Δ vs Last Close (%)": ((pred - last_close) / last_close) * 100,
            }
            for name, pred in predictions.items()
        ]
    ).sort_values(by="Predicted Close ($)")
    table["Predicted Close ($)"] = table["Predicted Close ($)"].map(format_currency)
    table["Δ vs Last Close ($)"] = table["Δ vs Last Close ($)"].map(format_currency)
    table["Δ vs Last Close (%)"] = table["Δ vs Last Close (%)"].map(lambda x: f"{x:+.2f}%")
    return table.reset_index(drop=True)


def add_timezone_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    utc_series = pd.to_datetime(enriched["merge_date"], utc=True, errors="coerce")
    enriched["date_utc"] = utc_series.dt.tz_convert("UTC")
    enriched["date_pst"] = utc_series.dt.tz_convert(TIMEZONE_CHOICES["Pacific (US)"])
    return enriched


def format_timestamp_column(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m-%d %H:%M %Z")


def render_history_chart(
    df: pd.DataFrame, prediction_value: float, lookback: int, timezone_label: str
):
    if "merge_date" not in df.columns or "close" not in df.columns:
        return None
    tz_name = TIMEZONE_CHOICES.get(timezone_label, "UTC")
    history = df[["merge_date", "close"]].copy()
    times = pd.to_datetime(history["merge_date"], utc=True, errors="coerce")
    times = times.dt.tz_convert(tz_name).dt.tz_localize(None)
    history["timestamp"] = times
    history = history.dropna(subset=["timestamp", "close"]).tail(lookback)
    if history.empty:
        return None
    next_date = history["timestamp"].iloc[-1] + pd.Timedelta(days=1)
    predicted = pd.DataFrame({"timestamp": [next_date], "close": [prediction_value]})
    axis_title = f"Date ({timezone_label})"
    if alt:
        base = (
            alt.Chart(history)
            .mark_line(color="#1f77b4")
            .encode(
                x=alt.X("timestamp:T", title=axis_title),
                y="close:Q",
                tooltip=["timestamp:T", "close:Q"],
            )
        )
        forecast = (
            alt.Chart(predicted)
            .mark_point(color="#ff7f0e", size=90)
            .encode(
                x=alt.X("timestamp:T", title=axis_title),
                y="close:Q",
                tooltip=["timestamp:T", "close:Q"],
            )
        )
        return (base + forecast).properties(height=320)
    history["series"] = "History"
    predicted["series"] = "Prediction"
    expanded = pd.concat([history, predicted], ignore_index=True)
    expanded = expanded.set_index("timestamp")
    return expanded


def validate_paths(
    models_dir: str, quant: str, google: str, interest: str, ingest: bool
) -> List[str]:
    errors = []
    if not models_dir:
        errors.append("Models directory is required.")
    elif not Path(models_dir).exists():
        errors.append("Models directory does not exist.")
    if not ingest:
        for label, path in [
            ("Quant CSV", quant),
            ("News sentiment CSV", google),
            ("Interest rates CSV", interest),
        ]:
            if not path:
                errors.append(f"{label} is required.")
            elif not Path(path).exists():
                errors.append(f"{label} was not found at:\n{path}")
    return errors


load_dotenv()
api_key = os.getenv("gemini_api_key")

st.set_page_config(page_title="Bitcoin Price Forecast", layout="wide")
st.title("Bitcoin Price Forecast Dashboard")
st.markdown(
    "Run the full prediction pipeline, compare model forecasts, review recent price action, "
    "and optionally generate LLM commentary with live market context."
)

with st.sidebar:
    st.header("Settings")
    model_runs = list_runs(str(MODELS_ROOT))
    if model_runs:
        selected_model_run = st.selectbox("Model artifacts", model_runs, index=0)
    else:
        st.warning("No saved model runs detected under ./models")
        selected_model_run = ""
    default_model_dir = (
        str((MODELS_ROOT / selected_model_run).resolve())
        if selected_model_run
        else str(MODELS_ROOT.resolve())
    )
    models_directory = st.text_input("Models directory", value=default_model_dir)

    st.divider()
    data_runs = list_runs(str(DATA_ROOT))
    if data_runs:
        selected_data_run = st.selectbox("Data snapshot", data_runs, index=0)
    else:
        st.warning("No data snapshots found under ./data")
        selected_data_run = ""
    ingest_new_data = st.toggle(
        "Ingest fresh data (~10-15 min)",
        value=False,
        help="Skip manual CSV selection and pull a new 20-day snapshot via ccxt/news/interest APIs.",
    )
    if ingest_new_data:
        st.info("This will run live ingestion and may take up to 15 minutes. Keep the app open.")
    auto_quant, auto_google, auto_interest = (
        infer_data_paths(selected_data_run) if selected_data_run else ("", "", "")
    )
    quant_path = st.text_input("Quant CSV", value=auto_quant, disabled=ingest_new_data)
    google_path = st.text_input("News sentiment CSV", value=auto_google, disabled=ingest_new_data)
    interest_path = st.text_input("Interest rates CSV", value=auto_interest, disabled=ingest_new_data)

    st.divider()
    dataset_length = 0 if ingest_new_data else get_dataset_length(quant_path)
    if ingest_new_data:
        existing_results = st.session_state.get("forecast_results")
        if existing_results and isinstance(existing_results, dict):
            df_cached = existing_results.get("data")
            if df_cached is not None:
                dataset_length = len(df_cached)
    slider_max = dataset_length if dataset_length > 0 else DEFAULT_HISTORY_WINDOW
    slider_max = max(slider_max, 1)
    slider_min = 5 if slider_max >= 5 else 1
    slider_default = min(DEFAULT_HISTORY_WINDOW, slider_max)
    slider_default = slider_default if slider_default >= slider_min else slider_min
    history_window = st.slider(
        "History window (days)",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_default,
        step=1,
        help="Limits the trailing window used for the price chart.",
    )
    timezone_label = st.radio(
        "Display timezone",
        list(TIMEZONE_CHOICES.keys()),
        index=0,
        help="Switch between UTC and Pacific (US); daylight savings handled automatically.",
    )
    include_commentary = st.toggle("Include AI prediction and commentary", value=bool(api_key))
    run_button = st.button("Run forecast", type="primary", width="stretch")
    st.caption("Selections auto-populate from ./models and ./data; override as needed.")

if "forecast_results" not in st.session_state:
    st.session_state["forecast_results"] = None

client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        st.sidebar.error(f"Gemini client unavailable: {exc}")
        client = None
elif include_commentary:
    st.sidebar.info("Set GEMINI_API_KEY in .env to enable commentary.")
    include_commentary = False

if run_button:
    issues = validate_paths(models_directory, quant_path, google_path, interest_path, ingest_new_data)
    if issues:
        for issue in issues:
            st.error(issue)
    else:
        with st.spinner("Running prediction pipeline..."):
            try:
                df, predictions = run_prediction_pipeline(
                    models_dir=models_directory.strip(),
                    quant_path=None if ingest_new_data else quant_path.strip(),
                    google_path=None if ingest_new_data else google_path.strip(),
                    interest_path=None if ingest_new_data else interest_path.strip(),
                    ingest=ingest_new_data,
                    save_dir=str(DATA_ROOT),
                )
            except Exception as exc:
                st.error(f"Prediction pipeline failed: {exc}")
                predictions = None
                df = None
        if predictions:
            st.session_state["forecast_results"] = {
                "data": df,
                "predictions": predictions,
                "history_window": history_window,
            }
            st.success("Predictions generated.")

results = st.session_state.get("forecast_results")
if results:
    df = results["data"]
    predictions = results["predictions"]
    history_window = results.get("history_window", DEFAULT_HISTORY_WINDOW)

    if df is None or not isinstance(predictions, dict):
        st.info("Run the forecast to view results.")
    else:
        history_window = min(history_window, len(df))
        df_with_tz = add_timezone_columns(df)
        last_close = float(df["close"].iloc[-1])
        pred_series = pd.Series(predictions)
        avg_pred = float(pred_series.mean())
        spread = float(pred_series.max() - pred_series.min())
        pct_delta = ((avg_pred - last_close) / last_close) * 100
        lr_prediction = float(predictions.get("Linear Regression", avg_pred))

        metric_cols = st.columns(3)
        metric_cols[0].metric("Latest close", format_currency(last_close))
        metric_cols[1].metric("Avg prediction", format_currency(avg_pred), f"{pct_delta:+.2f}% vs last close")
        metric_cols[2].metric("Model spread", format_currency(spread))

        chart = render_history_chart(df, lr_prediction, history_window, timezone_label)
        if chart is not None:
            if alt and isinstance(chart, alt.TopLevelMixin):
                st.altair_chart(chart, width="stretch")
            else:
                st.line_chart(chart)

        st.subheader("Model predictions")
        pred_table = build_predictions_table(predictions, last_close)
        st.dataframe(pred_table, width="stretch")
        st.download_button(
            label="Download predictions as CSV",
            data=pred_table.to_csv(index=False).encode("utf-8"),
            file_name="bitcoin_predictions.csv",
        )

        df_preview = df_with_tz.copy()
        for col in ("date_utc", "date_pst"):
            if col in df_preview.columns and pd.api.types.is_datetime64_any_dtype(df_preview[col]):
                df_preview[col] = format_timestamp_column(df_preview[col])
        tz_cols = [c for c in ("date_utc", "date_pst") if c in df_preview.columns]
        ordered_cols = tz_cols + [c for c in df_preview.columns if c not in tz_cols]
        with st.expander("Preview engineered features (last 15 rows)", expanded=False):
            st.dataframe(df_preview[ordered_cols].tail(15), width="stretch")

        if include_commentary:
            if not client:
                st.info("Gemini client unavailable; commentary skipped.")
            else:
                st.subheader("AI forecast commentary")
                prompt = get_prompt(predictions, df["close"].iloc[-1])
                tools_list = [
                    {"url_context": {}},
                    types.Tool(google_search=types.GoogleSearch()),
                ]
                urls_list = urls()
                try:
                    with st.spinner("Generating AI commentary..."):
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[prompt + f" Use these URLs for context: {urls_list}"],
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                                temperature=0.0,
                                tools=tools_list,
                            ),
                        )
                    st.markdown(add_citations(response))
                except Exception as exc:
                    st.error(f"Failed to generate commentary: {exc}")
else:
    st.info("Select a model/data snapshot and click Run forecast to begin.")
