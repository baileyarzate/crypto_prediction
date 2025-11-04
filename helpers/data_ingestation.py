import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import pytz
from tqdm import tqdm
import requests
import os
from google_news_api import GoogleNewsClient
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from helpers.queries import crypto_queries_with_limits

# --- Paths for saving data ---
# Use relative paths from the project root for portability
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')

def get_quant_data(save = True):
    print("--- Fetching Quant Data ---")
    exchange = ccxt.coinbase()
    symbol = 'BTC/USD'
    timeframe='1d'
    pst = pytz.timezone('America/Los_Angeles')

    now_ms = int(datetime.utcnow().timestamp() * 1000)
    since_ms = now_ms - 365 * 24 * 60 * 60 * 1000  # 1 year ago in ms

    all_data = []

    # Loop until now
    pbar = tqdm(total=now_ms - since_ms, unit='ms', desc="Fetching daily OHLCV")
    while since_ms < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
            if not ohlcv:
                break
            all_data += ohlcv
            # Advance to the next batch
            since_ms = ohlcv[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            pbar.update(ohlcv[-1][0] - ohlcv[0][0])
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print("Error:", e)
            time.sleep(5)
    pbar.close()
    df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
    df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    if save:
        timestamp_str = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d_%H%M")
        filename = os.path.join(SAVE_DIR, f'quant_bitcoin_test_{timestamp_str}.csv')
        df.to_csv(filename, index=False)
        print(f"Quant data saved to: {filename}")
    return filename

def get_interest_data(save = True):
    # Your interest rate function, modified to save and return the filename
    print("--- Fetching Interest Rate Data ---")
    # Calculate the date one year ago for the filter (e.g., '2024-10-30')
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # --- API Variables for Interest Rates (Using v2 from documentation) ---
    baseUrl = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service'
    # *** CORRECTED ENDPOINT based on the documentation provided ***
    endpoint = '/v2/accounting/od/avg_interest_rates' 
    API_URL = f'{baseUrl}{endpoint}'

    # Define parameters in a dictionary for proper URL encoding
    params = {
        'fields': 'record_date,security_type_desc,security_desc,avg_interest_rate_amt',
        'filter': f'record_date:gte:{one_year_ago}', # Filter for the last year
        'sort': 'record_date', 
        'format': 'json',
        'page[size]': 500  # Passed as a dictionary, requests handles the encoding
    }

    print(f"Attempting to call API at base URL: {API_URL}")

    # Call API and load into a pandas dataframe
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Check if 'data' key exists and is not empty
        if 'data' in data and data['data']:
            # This line should be inside the 'if'
            df = pd.DataFrame(data['data']) 
            print("\n🎉 API Call Successful!")
        else:
            print("No data found for the specified filter.")
            df = pd.DataFrame() 
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        df = pd.DataFrame()

    # This second definition is redundant and can be removed
    # df = pd.DataFrame(data['data']) 
    
    # Save whatever df was created (even if it's empty)
    if save:
        timestamp_str = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d_%H%M")
        filename = os.path.join(SAVE_DIR, f'interest_rates_test_{timestamp_str}.csv')
        df.to_csv(filename, index=False)
        print(f"Interest rate data saved to: {filename}")
    return filename

def extract_google_sentiment(hours = 400, save = True):
    tokenizer = BertTokenizer.from_pretrained("kk08/CryptoBERT")
    model = BertForSequenceClassification.from_pretrained("kk08/CryptoBERT")

    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Initialize the client
    client = GoogleNewsClient(language="en", country="US")

    def clean_html(raw_html):
        return BeautifulSoup(raw_html, "html.parser").get_text(strip=True)

    def get_dynamic_dates(hours_ago: int = 12):
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours_ago)
        return start_time.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H:%M:%SZ")

    def fetch_articles(query: str, hours_ago: int, max_results: int = 20):
        after, before = get_dynamic_dates(hours_ago)
        articles = client.search(
            query,
            after=after[:10],  # API accepts YYYY-MM-DD
            before=before[:10],
            max_results=max_results
        )
        # Clean and structure the results
        data = []
        for a in articles:
            title = clean_html(a.get("title", ""))
            summary = clean_html(a.get("summary", ""))
            published = a.get("published", "")
            data.append([query, hours_ago, title, summary, published])

        return pd.DataFrame(data, columns=["query", 'time_period', "title", "summary", "published"])

    def _apply_sentiment_batch(df, batch_size=32):
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
            elif label == "LABEL_0":
                return score
            else:
                return 0

        df['title_sentiment'] = [signed_score(r) for r in title_results]
        df['summary_sentiment'] = [signed_score(r) for r in summary_results]

        df['post'] = df['title'] + ' ' + df['summary']
        df['weighted_sentiment'] = 0.7 * df['title_sentiment'] + 0.3 * df['summary_sentiment']
        df = df.drop(columns=['title', 'summary', 'title_sentiment', 'summary_sentiment'])

        return df

    dfs = []
    for q, max_articles in tqdm(crypto_queries_with_limits(scale=int(hours/12)), desc='extracting articles from queries...'):
        dfs.append(fetch_articles(q, hours, max_results=max_articles))
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['title', 'summary'])

    # Apply sentiment with tqdm
    tqdm.pandas(desc="Applying sentiment analysis")
    df = _apply_sentiment_batch(df)

    if save:
        timestamp_str = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d_%H%M")
        filename = os.path.join(SAVE_DIR, f'google_news_sentiment_test_{timestamp_str}_hours_{hours}.csv')
        df.to_csv(filename, index=False)
        print(f"Google sentiment data saved to: {filename}")
    return filename
