import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from utils.indicators import add_indicators
from utils.model_utils import create_sequences, forecast_future_days

st.set_page_config(page_title='Stock Price Predictor', layout='wide')

st.title("📈 Stock Price Predictor with LSTM")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
period = st.selectbox("Select time period", ["1y", "2y", "5y"])
n_forecast = st.slider("Days to forecast", 1, 30, 7)
predict_button = st.button("Predict")

if predict_button:
    # Load data
    df = yf.download(ticker, period=period)
    df = df[['Close']].dropna()
    df = add_indicators(df)

    st.subheader("📊 Stock Closing Price")
    st.line_chart(df[['Close']])

    # Normalize and create sequences
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])

    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - 60:]

    X_train, y_train = create_sequences(train_data, 60)
    X_test, y_test = create_sequences(test_data, 60)

    # Show train/test split
    st.subheader("🧪 Train/Test Split")
    fig, ax = plt.subplots()
    ax.plot(scaled_data[:train_len], label='Train')
    ax.plot(range(train_len, len(scaled_data)), scaled_data[train_len:], label='Test')
    ax.legend()
    st.pyplot(fig)

    # Load or train model
    model_path = f"models/{ticker}.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        model.save(model_path)

    # Prediction
    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    st.subheader("📈 Predicted vs Actual")
    fig2, ax2 = plt.subplots()
    ax2.plot(actual_prices, label='Actual')
    ax2.plot(predicted_prices, label='Predicted')
    ax2.legend()
    st.pyplot(fig2)

    # Forecast future
    future_pred = forecast_future_days(model, scaled_data, n_forecast, scaler)
    st.subheader(f"🔮 Next {n_forecast} Day Forecast")
    st.line_chart(future_pred)

    # Display indicators
    st.subheader("📉 Technical Indicators")
    st.line_chart(df[['RSI', 'MACD']])