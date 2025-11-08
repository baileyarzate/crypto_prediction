def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, 1e-6)
    return 100 - (100 / (1 + rs))

def feature_engineering(df):
    import numpy as np
    """
    Takes a preprocessed DataFrame and creates all necessary time-series features.
    """
    print("--- Performing Feature Engineering ---")
    if len(df) < 15:
        print('Dataframe size is less than 15, get more data.')
    df_feat = df.copy()
    df_feat['rsi_14'] = calculate_rsi(df_feat, window=14)
    df_feat['lag_1'] = df_feat['close'].shift(1)
    df_feat['lag_5'] = df_feat['close'].shift(5)
    df_feat['lag_10'] = df_feat['close'].shift(10)
    df_feat['rolling_mean_5'] = df_feat['close'].rolling(window=5).mean()
    df_feat['rolling_std_5'] = df_feat['close'].rolling(window=5).std()
    df_feat['rolling_mean_10'] = df_feat['close'].rolling(window=10).mean()
    df_feat['rolling_std_10'] = df_feat['close'].rolling(window=10).std()
    log_return = np.log(df_feat['close'] / df_feat['close'].shift(1))
    df_feat['volatility_7'] = log_return.rolling(window=7).std() # 7-day volatility
    df_feat['momentum_5'] = log_return.rolling(window=5).mean() # 5-day momentum (average return)
    df_feat['rsi_14'] = calculate_rsi(df_feat, window=14)
    df_feat['day_of_week'] = df_feat['merge_date'].dt.dayofweek
    df_feat['high_low_spread'] = df_feat['high'] - df_feat['low']
    df_feat['momentum_x_volume'] = df_feat['momentum_5'] * df_feat['volume']
    df_feat['rsi_sq'] = df_feat['rsi_14']**2
    
    return df_feat