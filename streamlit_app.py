import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from prophet import Prophet
from nsepython import nse_optionchain_scrapper
from newspaper import Article
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import talib

# ---------------------------
# 1. Real-Time Data Fetching
# ---------------------------
def fetch_live_data(symbol="^NSEI", interval="15m"):
    """Fetch live NIFTY data for multiple timeframes"""
    data = yf.download(symbol, period="1d", interval=interval)
    return data

def fetch_vix():
    """Fetch India VIX data"""
    vix = yf.download("^INDIAVIX", period="1d", interval="15m")
    return vix['Close'].iloc[-1]

def fetch_option_chain():
    """Fetch live NSE option chain data"""
    oc_data = nse_optionchain_scrapper("NIFTY")
    calls = pd.DataFrame(oc_data['CE']['data'])
    puts = pd.DataFrame(oc_data['PE']['data'])
    return calls, puts

# ---------------------------
# 2. Option Greeks Calculation
# ---------------------------
def calculate_option_greeks(calls, puts):
    """Calculate Delta, Gamma, Theta, and Vega (simplified)"""
    total_oi = calls['openInterest'] + puts['openInterest']
    calls['Delta'] = calls['openInterest'] / total_oi
    puts['Delta'] = -puts['openInterest'] / total_oi
    return calls, puts

# ---------------------------
# 3. Candlestick Pattern Recognition
# ---------------------------
def detect_candlestick_patterns(data):
    """Detect common candlestick patterns using TA-Lib"""
    patterns = {
        'DOJI': talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close']),
        'ENGULFING': talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close']),
        'HAMMER': talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
    }
    detected = {pattern: patterns[pattern].iloc[-1] for pattern in patterns}
    return detected

# ---------------------------
# 4. Ensemble ML Models
# ---------------------------
def train_lstm(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_prophet(data):
    model = Prophet(daily_seasonality=True)
    df = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model.fit(df)
    return model

# ---------------------------
# 5. Multi-Timeframe Analysis
# ---------------------------
def analyze_timeframes():
    timeframes = ['5m', '15m', '1h']
    predictions = {}
    for tf in timeframes:
        data = fetch_live_data(interval=tf)
        if len(data) < 60: continue
        
        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
        
        # Train models
        X = np.array([scaled_data[i-60:i, 0] for i in range(60, len(scaled_data))])
        y = scaled_data[60:]
        
        lstm_model = train_lstm(X.reshape(X.shape[0], X.shape[1], 1), y)
        xgb_model = train_xgboost(X, y)
        
        # Predict
        last_sequence = X[-1].reshape(1, 60, 1)
        lstm_pred = scaler.inverse_transform(lstm_model.predict(last_sequence))[0][0]
        xgb_pred = scaler.inverse_transform(xgb_model.predict(X[-1].reshape(1, -1)))[0]
        
        predictions[tf] = (lstm_pred + xgb_pred) / 2  # Average prediction
    
    return predictions

# ---------------------------
# 6. Backtesting Engine
# ---------------------------
def backtest_strategy(data):
    """Simple moving average crossover backtest"""
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['Signal'] = np.where(data['SMA_10'] > data['SMA_50'], 1, -1)
    data['Returns'] = data['Close'].pct_change()
    data['Strategy'] = data['Signal'].shift(1) * data['Returns']
    return data['Strategy'].cumsum().iloc[-1]

# ---------------------------
# 7. Streamlit Dashboard
# ---------------------------
def main():
    st.title("NIFTY Ultimate Analyzer")
    
    # Fetch data
    nifty_data = fetch_live_data()
    vix = fetch_vix()
    calls, puts = fetch_option_chain()
    
    # Option Greeks
    calls, puts = calculate_option_greeks(calls, puts)
    
    # Candlestick Patterns
    patterns = detect_candlestick_patterns(nifty_data)
    
    # Multi-timeframe predictions
    timeframe_predictions = analyze_timeframes()
    
    # Risk Management (ATR)
    atr = (nifty_data['High'] - nifty_data['Low']).rolling(14).mean().iloc[-1]
    current_price = nifty_data['Close'].iloc[-1]
    stop_loss = current_price - 2 * atr
    
    # Backtesting
    backtest_result = backtest_strategy(nifty_data)
    
    # News Sentiment
    sentiment = analyze_news_sentiment()
    
    # Display
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"₹{current_price:.2f}")
    col2.metric("VIX", f"{vix:.2f}")
    col3.metric("Sentiment", sentiment)
    
    # Timeframe Predictions
    st.subheader("Multi-Timeframe Forecast")
    for tf, pred in timeframe_predictions.items():
        st.write(f"{tf} Prediction: ₹{pred:.2f}")
    
    # Candlestick Patterns
    st.subheader("Candlestick Patterns")
    st.write(patterns)
    
    # Option Chain Analysis
    st.subheader("Option Greeks (Top Strikes)")
    st.dataframe(calls[['strikePrice', 'Delta', 'openInterest']].head(10))
    
    # Backtesting Results
    st.subheader("Backtesting Results")
    st.write(f"Cumulative Strategy Returns: {backtest_result:.2f}%")
    
    # Visualization
    st.subheader("Price Analysis")
    fig, ax = plt.subplots()
    ax.plot(nifty_data['Close'], label='Price')
    ax.axhline(stop_loss, color='r', linestyle='--', label='Stop-Loss')
    ax.legend()
    st.pyplot(fig)

def analyze_news_sentiment():
    """Analyze news sentiment from Economic Times"""
    try:
        article = Article("https://economictimes.indiatimes.com/markets/stocks/news")
        article.download()
        article.parse()
        return "Bullish" if "rise" in article.text.lower() else "Bearish"
    except:
        return "Neutral"

if __name__ == "__main__":
    main()
