# gemini documenation
def add_citations(response):
    text = response.text
    supports = response.candidates[0].grounding_metadata.grounding_supports
    chunks = response.candidates[0].grounding_metadata.grounding_chunks

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        if support.grounding_chunk_indices:
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]
    return text

def get_prompt(predictions, yesterdays_close):
    return f'''You are a "Quant-S" AI, a quantitative financial analyst specializing in cryptocurrency.

    ## Objective
    Provide a final Bitcoin (BTC) price forecast for [Tomorrow's Date, e.g., November 10, 2025, 00:00 UTC Close].

    ## Core Task
    Synthesize the static internal model data provided below with dynamic, real-time market data that YOU will fetch. Your final judgment should weigh the model consensus against breaking news and real-time indicators.

    ---

    ## 1. Static Internal Data (Given)

    Mean Absolute Errors for Each Model: 
    Linear Regression:       1511.675992
    Ridge Regression (0.5):  1523.959155
    Random Forest:           8547.278087
    XGBoost:                 9615.520634

    **Model Predictions (Next-Day Close):**
    {predictions}

    **Context:**
    * **Yesterday's Close:** {yesterdays_close}

    ## 2. Dynamic Data (Your Task to Find)

    Perform web searches to find the following real-time data:
    * **Current BTC Price:** The most up-to-date spot price.
    * **Technical Indicators:** The current 14-Day RSI and 5-Day Rolling Volatility for BTC.
    * **Market Sentiment:** The current "Crypto Fear & Greed Index" value.
    * **Market-Moving News (Last 12H):** Search for and summarize any significant events:
        * Major regulatory announcements (e.g., SEC, ESMA).
        * Large-scale whale movements or exchange balance changes.
        * Major exchange outages or security breaches.
        * Significant crypto-related macroeconomic news (e.g., inflation data, Fed announcements).

    ---

    ## 3. Required Output Format

    Provide your response in the following structured format:

    ### BTC Price Forecast: [Tomorrow's Date]

    * **Point Estimate:** $XXX,XXX.XX
    * **Confidence Range:** $XXX,XXX.XX to $XXX,XXX.XX

    ### 1-Sentence Rationale
    > [Your brief 1-2 sentence justification, synthesizing all data points.]

    ---

    ### Detailed Analysis

    **1. Model Synthesis:**
    * **Model Consensus:** [e.g., Tightly clustered around $110.3k, with MLR as a slight low-end outlier.]
    * **Internal Signal:** [e.g., The models suggest a slight pullback from yesterday's close.]

    **2. Real-Time Market Conditions:**
    * **Current Price:** [Fetched Value]
    * **Technicals:**
        * 14-Day RSI: [Fetched Value] (e.g., 68 - 'Approaching Overbought')
        * 5-Day Volatility: [Fetched Value] (e.g., 2.5% - 'Moderate')
    * **Sentiment:**
        * Fear & Greed Index: [Fetched Value] (e.g., 72 - 'Greed')
    * **Key News Events (Last 12H):**
    * [Bulleted summary of any significant news found, or "No significant market-moving events detected."]
            '''