def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, 1e-6)
    return 100 - (100 / (1 + rs))

def feature_engineering(df):
    from numpy import log, abs
    print("--- Performing Feature Engineering ---")
    if len(df) < 15:
        raise ValueError(f"Dataframe size too small ({len(df)} rows).")

    df_feat = df.copy()

    # Core technicals
    df_feat['rsi_14'] = calculate_rsi(df_feat, window=14)
    df_feat['lag_1'] = df_feat['close'].shift(1)
    df_feat['lag_5'] = df_feat['close'].shift(5)
    df_feat['lag_10'] = df_feat['close'].shift(10)
    df_feat['rolling_mean_5'] = df_feat['close'].rolling(window=5).mean()
    df_feat['rolling_std_5'] = df_feat['close'].rolling(window=5).std()
    df_feat['rolling_mean_10'] = df_feat['close'].rolling(window=10).mean()
    df_feat['rolling_std_10'] = df_feat['close'].rolling(window=10).std()

    # Log return volatility / momentum
    log_return = log(df_feat['close'] / df_feat['close'].shift(1))
    df_feat['volatility_7'] = log_return.rolling(window=7).std()
    df_feat['momentum_5'] = log_return.rolling(window=5).mean()

    # Derived interaction features
    df_feat['high_low_spread'] = df_feat['high'] - df_feat['low']
    df_feat['momentum_x_volume'] = df_feat['momentum_5'] * df_feat['volume']
    df_feat['rsi_sq'] = df_feat['rsi_14'] ** 2
    df_feat['volatility_7_sq'] = df_feat['volatility_7'] ** 2

    # Delta / deviation features
    df_feat['close_delta_5'] = df_feat['close'] - df_feat['close'].shift(5)
    df_feat['log_return_abs'] = abs(log_return)
    df_feat['high_low_vol_ratio'] = df_feat['high_low_spread'] / (df_feat['rolling_std_10'] + 1e-6)

    # Adaptive rolling signals (EWMA)
    df_feat['ewma_5'] = df_feat['close'].ewm(span=5, adjust=False).mean()
    df_feat['ewma_10'] = df_feat['close'].ewm(span=10, adjust=False).mean()
    df_feat['ewma_ratio'] = df_feat['ewma_5'] / (df_feat['ewma_10'] + 1e-6)

    # Day of week effect
    df_feat['day_of_week'] = df_feat['merge_date'].dt.dayofweek

    return df_feat