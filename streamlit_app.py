# streamlit_app.py
import streamlit as st
import pandas as pd
from main import run_prediction_pipeline
from helpers.llm_support import get_prompt, add_citations
from helpers.queries import urls
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)

# --- Streamlit UI ---
st.set_page_config(page_title="Bitcoin Price Forecast", layout="wide")
st.title("Bitcoin Price Forecast Dashboard")
st.markdown(
    "This dashboard displays predictions for tomorrow's Bitcoin closing price "
    "from multiple models, along with AI-generated commentary."
)

# --- Sidebar options ---
st.sidebar.header("Settings")
models_directory = st.sidebar.text_input(
    "Models Directory",
    value=r"C:\Users\baile\Documents\Artificial Intelligence\BitcoinPred\models\20251112_170018"
)
run_button = st.sidebar.button("Run Forecast")

if run_button:
    with st.spinner("Running prediction pipeline..."):
        # Run your backend prediction pipeline
        df, predictions = run_prediction_pipeline(quant_path=r'C:\Users\baile\Documents\Artificial Intelligence\BitcoinPred\data\20251112_172623\quant\quant_bitcoin_test_20251112_1726.csv',
                                                  google_path=r'C:\Users\baile\Documents\Artificial Intelligence\BitcoinPred\data\20251112_172623\sentiment\google_news_sentiment_20251112_1739_days_20.csv',
                                                  interest_path=r'C:\Users\baile\Documents\Artificial Intelligence\BitcoinPred\data\20251112_172623\interest\interest_rates_test_20251112_1739.csv',
                                                  models_dir=models_directory)

    st.success("Predictions generated!")
    
    # Display prediction table
    st.subheader("Model Predictions")
    pred_df = pd.DataFrame(
    predictions.items(),
    columns=['Model Name', 'Predicted Close Price ($)']
    )

    # Set the 'Model Name' as the index for a cleaner look
    pred_df = pred_df.set_index('Model Name')

    # Format the price to two decimal places
    pred_df['Predicted Close Price ($)'] = pred_df['Predicted Close Price ($)'].map('{:,.2f}'.format)

    st.dataframe(pred_df)
    
    # Generate AI commentary
    st.subheader("AI Forecast Commentary")
    prompt = get_prompt(predictions, df['close'].iloc[-1])
    
    # Build the tools list
    tools_list = [
        {"url_context": {}},  # placeholder for user-defined URLs
        types.Tool(google_search=types.GoogleSearch())
    ]
    
    # Get URLs from helper
    urls_list = urls()
    
    # Generate AI content
    with st.spinner("Generating AI commentary..."):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt + f" Look through these and some of your own URLs not listed here: {urls_list}"],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                temperature=0.0,
                tools=tools_list
            )
        )
        
        text_with_citations = add_citations(response)
    
    st.markdown(text_with_citations)
