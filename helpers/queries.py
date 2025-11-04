def crypto_queries_with_limits(scale = 1):
    base_queries = [
        # --- Major coins (high priority) ---
        ("Bitcoin", 20),
        ("BTC", 20),
        ("Ethereum", 18),
        ("ETH", 18),
        ("Binance Coin", 12),
        ("BNB", 12),
        ("Cardano", 10),
        ("ADA", 10),
        ("Solana", 10),
        ("SOL", 10),

        # --- Key DeFi & Web3 ---
        ("DeFi", 12),
        ("Web3", 12),
        ("NFT", 12),
        ("NFT marketplace", 10),
        ("Metaverse", 10),
        ("Yield farming", 8),
        ("Liquidity pools", 8),
        ("Staking crypto", 8),

        # --- Exchanges & trading platforms ---
        ("Binance", 10),
        ("Coinbase", 10),
        ("Kraken", 8),
        ("Bybit", 8),
        ("OKX", 8),

        # --- Market events & trends ---
        ("Bitcoin halving", 6),
        ("Ethereum merge", 6),
        ("Crypto bull run", 6),
        ("Crypto crash", 6),
        ("Altcoin season", 6),

        # --- Influencers & social sentiment ---
        ("Elon Musk crypto", 6),
        ("Vitalik Buterin", 6),
        ("CZ Binance", 6),
        ("Crypto Twitter", 6),
        ("Reddit cryptocurrency", 6),

        # --- Misc / general news ---
        ("Crypto news", 10),
        ("Crypto market news", 10),
        ("Digital currency", 6),
        ("Virtual currency", 6),
        ("Crypto adoption", 6),
    ]
    # Apply scale to article limits
    scaled = [(query, int(limit * scale)) for query, limit in base_queries]
    return scaled

def subreddit_queries():
    return [
        # --- General Crypto ---
        "CryptoCurrency",
        "cryptomarkets",
        "CryptoNews",
        "CryptocurrencyTrading",
        
        # --- Bitcoin ---
        "Bitcoin",
        "btc",
        "BitcoinMarkets",
        
        # --- Ethereum ---
        "ethereum",
        "ethtrader",
        
        # --- Altcoins & Major Projects ---
        "altcoin",
        "Solana",
        "Cardano",
        "dogecoin",
        "Chainlink",
        
        # --- DeFi & Platforms ---
        "DeFi",
        
        # --- NFTs & Metaverse ---
        "NFT",
        
        # --- Exchanges & Trading Platforms ---
        "Binance",
        "Coinbase",
        "FTX",
        
        # --- Influencers ---
        "elonmusk",
    ]