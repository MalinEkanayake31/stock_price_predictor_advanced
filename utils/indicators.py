import ta

def add_indicators(df):
    close_series = df['Close'].squeeze()  # Ensure it's 1D
    df['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()
    df['MACD'] = ta.trend.MACD(close=close_series).macd()
    return df
