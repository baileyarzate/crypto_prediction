import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import pytz
from tqdm import tqdm
import requests
import os
from typing import Optional, Tuple, Iterable
from google_news_api import GoogleNewsClient
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from helpers.queries import crypto_queries_with_limits

"""
Flexible data ingestion utilities for:
- Quant price data (via ccxt)
- US Treasury interest rates
- Google News + CryptoBERT sentiment

Backwards compatible defaults: functions still return a file path string by
default (save=True, return_df=False). To get DataFrames in-memory, pass
return_df=True. If you set save=False, you must set return_df=True.
"""

# --- Paths for saving data ---
# Use relative paths from the project root for portability
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
SAVE_DIR = os.environ.get('SAVE_DIR', DEFAULT_SAVE_DIR)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_quant_data(
    save: bool = True,
    save_dir: Optional[str] = None,
    exchange_name: str = 'coinbase',
    symbol: str = 'BTC/USD',
    timeframe: str = '1d',
    lookback_days: int = 1095,
    return_df: bool = False,
    show_progress: bool = True,
):
    """
    Fetch OHLCV data via ccxt with configurable exchange/symbol/timeframe.

    Returns (by flags):
    - save=True,  return_df=False -> file path (str)
    - save=False, return_df=True  -> DataFrame
    - save=True,  return_df=True  -> (DataFrame, file path)
    - save=False, return_df=False -> ValueError
    """
    print("--- Fetching Quant Data ---")

    # Resolve exchange dynamically
    try:
        exchange_cls = getattr(ccxt, exchange_name)
        exchange = exchange_cls()
    except AttributeError:
        raise ValueError(f"Unknown exchange '{exchange_name}' for ccxt")

    now_ms = exchange.milliseconds()#int(datetime.utcnow().timestamp() * 1000)
    since_ms = now_ms - lookback_days * 24 * 60 * 60 * 1000

    all_data = []

    total_span = max(1, now_ms - since_ms)
    pbar = tqdm(total=total_span, unit='ms', desc="Fetching OHLCV", disable=not show_progress)
    while since_ms < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
            if not ohlcv:
                break
            all_data += ohlcv
            # Advance to the next batch
            since_ms = ohlcv[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            pbar.update(max(0, ohlcv[-1][0] - ohlcv[0][0]))
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print("Error:", e)
    pbar.close()

    df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
    if not df.empty:
        df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    if not save and not return_df:
        raise ValueError("Nothing to return: set save=True or return_df=True")

    out_path: Optional[str] = None
    if save:
        out_dir = save_dir or SAVE_DIR
        _ensure_dir(out_dir)
        timestamp_str = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d_%H%M")
        out_path = os.path.join(out_dir, f'quant_bitcoin_test_{timestamp_str}.csv')
        df.to_csv(out_path, index=False)
        print(f"Quant data saved to: {out_path}")

    if return_df and save:
        return df, out_path
    if return_df and not save:
        return df
    # save only
    return out_path

def _fetch_data_window(
    start_date: str, 
    end_date: str, 
    base_url: str, 
    endpoint: str, 
    params_override: Optional[dict] = None, 
    timeout: int = 30
) -> pd.DataFrame:
    """Fetches data for a single date window."""
    api_url = f'{base_url}{endpoint}'
    initial_columns = ['record_date','security_type_desc','security_desc','avg_interest_rate_amt']
    
    params = {
        'fields': ','.join(initial_columns),
        # Filter for the specific window: start_date <= record_date < end_date
        'filter': f'record_date:gte:{start_date},record_date:lt:{end_date}',
        'sort': 'record_date', 
        'format': 'json',
        'page[size]': 10000 # Use max size to ensure full year coverage in one page
    }
    if params_override:
        params.update(params_override)

    print(f"  -> Fetching: {start_date} to {end_date}")

    all_data = []
    has_more_pages = True
    page_num = 1
    
    # Simple pagination check in case a single year exceeds 10000 records
    while has_more_pages:
        params['page[number]'] = page_num
        
        try:
            response = requests.get(api_url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            
            if 'data' in payload and payload['data']:
                all_data.extend(payload['data'])
                has_more_pages = 'next' in payload.get('links', {})
                page_num += 1
            else:
                has_more_pages = False
                
        except requests.exceptions.RequestException as e:
            #print(f"  -> ERROR fetching page {page_num}: {e}")
            has_more_pages = False
    
    if all_data:
        return pd.DataFrame(all_data)
    else:
        return pd.DataFrame(columns=initial_columns)

# --- Main Wrapper Function ---
def get_interest_data(
    save: bool = True,
    save_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    lookback_days: int = 1095, # Example: 1095 days = 3 years
    base_url: str = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service',
    endpoint: str = '/v2/accounting/od/avg_interest_rates',
    params_override: Optional[dict] = None,
    timeout: int = 30,
    return_df: bool = False,
):
    
    print("--- Fetching Interest Rate Data (Year-by-Year) ---")
    
    # Determine the end of the data (Today)
    end_date_dt = datetime.now().date()
    
    # Calculate how many full years we need to loop back
    years_to_fetch = lookback_days // 365
    
    all_dfs = []
    
    current_end = end_date_dt # Start with today's date

    # Loop backward year by year
    for i in range(years_to_fetch):
        # Calculate the start date of this 1-year window
        current_start = (current_end - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Format the end date for the API (exclusive)
        api_end_date = current_end.strftime('%Y-%m-%d')
        
        # --- Fetch Data for this Window ---
        df_chunk = _fetch_data_window(
            start_date=current_start,
            end_date=api_end_date,
            base_url=base_url,
            endpoint=endpoint,
            params_override=params_override,
            timeout=timeout
        )
        all_dfs.append(df_chunk)
        
        # Set the end date for the next iteration to the start date of the current one
        current_end = datetime.strptime(current_start, '%Y-%m-%d').date()

    # --- Concatenate and Finalize ---
    if not all_dfs:
        print("No data fetched in any year.")
        df = pd.DataFrame()
    else:
        # Concatenate all the yearly data chunks
        df = pd.concat(all_dfs, ignore_index=True)
        # Drop duplicates in case the API returned overlapping records at the window boundaries
        df.drop_duplicates(subset=['record_date', 'security_desc'], inplace=True)
        df.sort_values('record_date', inplace=True)
        print(f"\n🎉 Total Unique Records Fetched: {len(df)}")
        
    # --- Data Saving and Return Logic ---
    out_path: Optional[str] = None
    if save and not df.empty:
        # NOTE: Using placeholder logic for saving
        out_dir = save_dir or "data" 
        # _ensure_dir(out_dir) 
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M") 
        out_path = os.path.join(out_dir, f'interest_rates_test_{timestamp_str}.csv')
        df.to_csv(out_path, index=False)
        print(f"Interest rate data saved to: {out_path}")

    if return_df and save:
        return df, out_path
    if return_df and not save:
        return df
    return out_path

def extract_google_sentiment(
    hours: int = 24000,  # 1000 days * 24 hours
    save: bool = True,
    save_dir: Optional[str] = None,
    queries: Optional[Iterable[Tuple[str, int]]] = None,
    batch_size: int = 32,
    model_name: str = "kk08/CryptoBERT",
    tokenizer_name: Optional[str] = None,
    max_results_per_query: Optional[int] = 5,  # Default 5/day to meet 5/day total goal
    return_df: bool = True,
):
    """
    Fetches Google News sentiment by looping day-by-day for a historical period
    and applying a rate-limit delay after a number of API queries.

    The fetching strategy iterates over 24-hour windows to ensure
    even article distribution, aiming for ~5 articles/day.
    """
    # --- Load Sentiment Model ---
    try:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name or model_name, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"⚠️ Warning: sentiment model load failed: {e}. Proceeding without model.")
        sentiment_pipeline = None  # type: ignore

    # --- Initialize Google News Client ---
    try:
        client = GoogleNewsClient(language="en", country="US")
    except Exception as e:
        print(f"⚠️ Warning: GoogleNews client failed: {e}. Returning empty sentiment.")
        client = None  # type: ignore

    # --- Helpers ---
    def clean_html(raw_html: str) -> str:
        """Removes HTML tags from text."""
        return BeautifulSoup(raw_html, "html.parser").get_text(strip=True)

    def _apply_sentiment_batch(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """Applies sentiment analysis in batches and computes weighted score."""
        if df.empty:
            return pd.DataFrame(columns=["post", "weighted_sentiment"])

        df["post"] = df.apply(lambda row: f"{row['title']} {row['summary']}", axis=1)

        if sentiment_pipeline is None:
            df["weighted_sentiment"] = 0.0
            return df[["post", "weighted_sentiment"]]

        texts_title = df["title"].tolist()
        texts_summary = df["summary"].tolist()
        title_results, summary_results = [], []

        for i in tqdm(range(0, len(df), batch_size), desc="Batch sentiment"):
            batch_title = texts_title[i:i + batch_size]
            batch_summary = texts_summary[i:i + batch_size]
            title_results.extend(sentiment_pipeline(batch_title))  # type: ignore
            summary_results.extend(sentiment_pipeline(batch_summary))  # type: ignore

        def signed_score(result) -> float:
            label, score = result.get("label", ""), result.get("score", 0)
            if label == "LABEL_0":
                return -score
            elif label == "LABEL_1":
                return score
            return 0.0

        df["title_sentiment"] = [signed_score(r) for r in title_results]
        df["summary_sentiment"] = [signed_score(r) for r in summary_results]
        df["weighted_sentiment"] = (
            0.7 * df["title_sentiment"] + 0.3 * df["summary_sentiment"]
        )
        return df[['query','time_period','published','post','weighted_sentiment']]

    # --- Fetch Logic ---
    if client is None:
        df_empty = pd.DataFrame(columns=["query", "time_period", "title", "summary", "published"])
        return _apply_sentiment_batch(df_empty, batch_size)

    if queries is None:
        try:
            queries_list = [q for q, _ in crypto_queries_with_limits(scale=1)]
        except Exception:
            queries_list = ["Bitcoin"]
    else:
        queries_list = [q for q, _ in queries]

    total_days = max(hours // 24, 1)
    cap = max_results_per_query or 5
    rows = []
    api_call_count = 0
    now = datetime.utcnow()

    for day_offset in tqdm(range(total_days), desc=f"Fetching {total_days} days"):
        end_time = now - timedelta(days=day_offset)
        start_time = end_time - timedelta(days=1)
        after_date, before_date = start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d")

        for query in queries_list:
            # if api_call_count > 0 and api_call_count % 900 == 0:
            #     print(f"Rate limit hit at {api_call_count} calls. Sleeping 60s...")
            #     time.sleep(60)

            try:
                articles = client.search(query, after=after_date, before=before_date, max_results=cap)
                if hours * max_results_per_query > 20:
                    time.sleep(1.05)  # add a 1.05-second pause after each API call
                api_call_count += 1
                
            except Exception as e:
                print(f"⚠️ Warning: search failed for '{query}' ({after_date}): {e}")
                articles = []

            for a in articles:
                rows.append([
                    query,
                    f"{after_date} to {before_date}",
                    clean_html(a.get("title", "")),
                    clean_html(a.get("summary", "")),
                    a.get("published", ""),
                ])

    # --- Sentiment Application ---
    df = pd.DataFrame(rows, columns=["query", "time_period", "title", "summary", "published"]).drop_duplicates()

    tqdm.pandas(desc="Applying sentiment analysis")
    df_sentiment = _apply_sentiment_batch(df, batch_size)

    # --- Save / Return ---
    if not save and not return_df:
        print("Nothing to return: set save=True or return_df=True. Setting return_df to True")
        return_df = True

    out_path = None
    if save:
        out_dir = save_dir or SAVE_DIR
        _ensure_dir(out_dir)
        timestamp_str = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d_%H%M")
        out_path = os.path.join(out_dir, f"google_news_sentiment_{timestamp_str}_days_{total_days}.csv")
        df_sentiment.to_csv(out_path, index=False)
        print(f"✅ Google sentiment data saved to: {out_path}")

    if return_df and save:
        return df_sentiment, out_path
    if return_df:
        return df_sentiment
    return out_path

if __name__ == "__main__":
    # Minimal CLI for ad-hoc ingestion
    import argparse

    parser = argparse.ArgumentParser(description="Data ingestion utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Quant
    p_quant = sub.add_parser("quant", help="Fetch OHLCV via ccxt")
    p_quant.add_argument("--exchange", default="coinbase")
    p_quant.add_argument("--symbol", default="BTC/USD")
    p_quant.add_argument("--timeframe", default="1d")
    p_quant.add_argument("--lookback-days", type=int, default=1095)
    p_quant.add_argument("--save-dir", default=None)

    # Interest
    p_ir = sub.add_parser("interest", help="Fetch US Treasury interest rates")
    p_ir.add_argument("--start-date", default=None, help="YYYY-MM-DD; default uses lookback")
    p_ir.add_argument("--lookback-days", type=int, default=1095)
    p_ir.add_argument("--save-dir", default=None)

    # Google sentiment
    p_news = sub.add_parser("news", help="Fetch Google News sentiment for crypto")
    p_news.add_argument("--hours", type=int, default=26280)
    p_news.add_argument("--batch-size", type=int, default=32)
    p_news.add_argument("--model", default="kk08/CryptoBERT")
    p_news.add_argument("--max-results-per-query", type=int, default=1)
    p_news.add_argument("--save-dir", default=None)

    args = parser.parse_args()
    if args.cmd == "quant":
        path = get_quant_data(
            save=True,
            save_dir=args.save_dir,
            exchange_name=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
        )
        print(path)
    elif args.cmd == "interest":
        path = get_interest_data(
            save=True,
            save_dir=args.save_dir,
            start_date=args.start_date,
            lookback_days=args.lookback_days,
        )
        print(path)
    elif args.cmd == "news":
        path = extract_google_sentiment(
            hours=args.hours,
            save=True,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            model_name=args.model,
            max_results_per_query=args.max_results_per_query,
        )
        print(path)
