import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import pytz
from tqdm import tqdm
import requests
import os
from typing import Optional, Tuple, Iterable, List
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
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')
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

    now_ms = int(datetime.utcnow().timestamp() * 1000)
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
            time.sleep(5)
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

def get_interest_data(
    save: bool = True,
    save_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    lookback_days: int = 1095,
    base_url: str = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service',
    endpoint: str = '/v2/accounting/od/avg_interest_rates',
    params_override: Optional[dict] = None,
    timeout: int = 30,
    return_df: bool = False,
):
    print("--- Fetching Interest Rate Data ---")
    # Determine start_date window
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    # --- API Variables for Interest Rates (Using v2 from documentation) ---
    api_url = f'{base_url}{endpoint}'

    # Define parameters in a dictionary for proper URL encoding
    params = {
        'fields': 'record_date,security_type_desc,security_desc,avg_interest_rate_amt',
        'filter': f'record_date:gte:{start_date}',
        'sort': 'record_date', 
        'format': 'json',
        'page[size]': 500
    }
    if params_override:
        params.update(params_override)

    print(f"Attempting to call API at base URL: {api_url}")

    # Call API and load into a pandas dataframe
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        payload = response.json()
        
        # Check if 'data' key exists and is not empty
        if 'data' in payload and payload['data']:
            # This line should be inside the 'if'
            df = pd.DataFrame(payload['data']) 
            print("\n🎉 API Call Successful!")
        else:
            print("No data found for the specified filter.")
            df = pd.DataFrame() 
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        df = pd.DataFrame()

    # This second definition is redundant and can be removed
    # df = pd.DataFrame(data['data']) 
    
    # Enforce explicit return choice and save if requested
    if not save and not return_df:
        raise ValueError("Nothing to return: set save=True or return_df=True")

    out_path: Optional[str] = None
    if save:
        out_dir = save_dir or SAVE_DIR
        _ensure_dir(out_dir)
        timestamp_str = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d_%H%M")
        out_path = os.path.join(out_dir, f'interest_rates_test_{timestamp_str}.csv')
        df.to_csv(out_path, index=False)
        print(f"Interest rate data saved to: {out_path}")

    if return_df and save:
        return df, out_path
    if return_df and not save:
        return df
    return out_path

def extract_google_sentiment(
    hours: int = 26280,
    save: bool = True,
    save_dir: Optional[str] = None,
    queries: Optional[Iterable[Tuple[str, int]]] = None,
    batch_size: int = 32,
    model_name: str = "kk08/CryptoBERT",
    tokenizer_name: Optional[str] = None,
    max_results_per_query: Optional[int] = None,
    return_df: bool = False,
):
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Warning: sentiment model load failed: {e}. Proceeding without model; downstream will set zeros.")
        sentiment_pipeline = None  # type: ignore

    # Initialize the client
    try:
        client = GoogleNewsClient(language="en", country="US")
    except Exception as e:
        print(f"Warning: GoogleNews client failed: {e}. Returning empty sentiment; downstream will set zeros.")
        client = None  # type: ignore

    def clean_html(raw_html):
        return BeautifulSoup(raw_html, "html.parser").get_text(strip=True)

    def get_dynamic_dates(hours_ago: int = 12):
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours_ago)
        return start_time.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H:%M:%SZ")

    def fetch_articles(query: str, hours_ago: int, max_results: int = 20):
        if client is None:
            return pd.DataFrame(columns=["query", 'time_period', "title", "summary", "published"])  # type: ignore
        after, before = get_dynamic_dates(hours_ago)
        try:
            articles = client.search(
                query,
                after=after[:10],  # API accepts YYYY-MM-DD
                before=before[:10],
                max_results=max_results
            )
        except Exception as e:
            # API limit or other error; return empty for this query
            print(f"Warning: news search failed for '{query}': {e}. Skipping.")
            articles = []
        # Clean and structure the results
        data = []
        for a in articles:
            title = clean_html(a.get("title", ""))
            summary = clean_html(a.get("summary", ""))
            published = a.get("published", "")
            data.append([query, hours_ago, title, summary, published])

        return pd.DataFrame(data, columns=["query", 'time_period', "title", "summary", "published"])

    def _apply_sentiment_batch(df, batch_size=32):
        if df.empty:
            # Ensure expected columns exist even if empty
            df['post'] = pd.Series(dtype=str)
            df['weighted_sentiment'] = pd.Series(dtype=float)
            return df
        if sentiment_pipeline is None:
            # No model available; set zeros
            df['post'] = df['title'] + ' ' + df['summary']
            df['weighted_sentiment'] = 0.0
            return df.drop(columns=['title', 'summary'], errors='ignore')
        # Combine title and summary for batching if you want weighted later
        texts_title = df['title'].tolist()
        texts_summary = df['summary'].tolist()

        title_results, summary_results = [], []

        for i in tqdm(range(0, len(df), batch_size), desc="Batch sentiment"):
            batch_title = texts_title[i:i+batch_size]
            batch_summary = texts_summary[i:i+batch_size]

            # Run sentiment on batch
            title_batch_result = sentiment_pipeline(batch_title)
            summary_batch_result = sentiment_pipeline(batch_summary)

            title_results.extend(title_batch_result) # type: ignore
            summary_results.extend(summary_batch_result) # type: ignore

        # Helper for mapping label -> signed score
        def signed_score(result):
            label = result['label']
            score = result['score']
            if label == "LABEL_0":
                return -score
            elif label == "LABEL_1":
                return score
            else:
                return 0

        df['title_sentiment'] = [signed_score(r) for r in title_results]
        df['summary_sentiment'] = [signed_score(r) for r in summary_results]

        df['post'] = df['title'] + ' ' + df['summary']
        df['weighted_sentiment'] = 0.7 * df['title_sentiment'] + 0.3 * df['summary_sentiment']
        df = df.drop(columns=['title', 'summary', 'title_sentiment', 'summary_sentiment'])

        return df

    # Build queries list
    if queries is None:
        # Avoid exploding request volume; use modest scale and cap per-query results
        bounded_scale = 1
        queries = crypto_queries_with_limits(scale=bounded_scale)
    if max_results_per_query is not None:
        queries = [(q, min(limit, max_results_per_query)) for q, limit in queries]
    else:
        # Provide a safe default cap to reduce API pressure when hours is very large
        queries = [(q, min(limit, 20)) for q, limit in queries]

    dfs = []
    for q, max_articles in tqdm(queries, desc='extracting articles from queries...'):
        if max_articles <= 0:
            continue
        dfs.append(fetch_articles(q, hours, max_results=max_articles))
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['title', 'summary'])
    else:
        df = pd.DataFrame(columns=["query", 'time_period', "title", "summary", "published"])  # type: ignore

    # Apply sentiment with tqdm
    tqdm.pandas(desc="Applying sentiment analysis")
    df = _apply_sentiment_batch(df)

    if not save and not return_df:
        raise ValueError("Nothing to return: set save=True or return_df=True")

    out_path: Optional[str] = None
    if save:
        out_dir = save_dir or SAVE_DIR
        _ensure_dir(out_dir)
        timestamp_str = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d_%H%M")
        out_path = os.path.join(out_dir, f'google_news_sentiment_test_{timestamp_str}_hours_{hours}.csv')
        df.to_csv(out_path, index=False)
        print(f"Google sentiment data saved to: {out_path}")

    if return_df and save:
        return df, out_path
    if return_df and not save:
        return df
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
    p_news.add_argument("--max-results-per-query", type=int, default=None)
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
