import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# ---------------------------
# 1. Live Data Fetching
# ---------------------------
def fetch_live_data(symbols=["^NSEI", "^NSEBANK"], interval="15m"):
    """Fetch live stock/index data from Yahoo Finance"""
    try:
        data = yf.download(tickers=symbols, period="1d", interval=interval, group_by='ticker')
        return data
    except Exception as e:
        st.error(f"Error fetching live data: {str(e)}")
        return None

def fetch_vix():
    """Fetch India VIX data"""
    try:
        vix = yf.download("^INDIAVIX", period="1d", interval="15m")
        return vix['Close'].iloc[-1]
    except Exception:
        return "N/A"

# ---------------------------
# 2. Option Chain Scraping with Selenium
# ---------------------------
def fetch_option_chain(index="NIFTY"):
    """Fetch option chain from NSE website using Selenium"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    service = Service('/path/to/chromedriver')  # Set your ChromeDriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = f"https://www.nseindia.com/option-chain"
    driver.get(url)
    time.sleep(5)  # Wait for page to load

    try:
        calls, puts = [], []
        rows = driver.find_elements(By.XPATH, '//table[contains(@class, "option-chain-table")]//tr')

        for row in rows[1:]:
            cols = row.find_elements(By.TAG_NAME, 'td')
            if len(cols) < 11:
                continue

            call_oi = cols[0].text
            call_price = cols[1].text
            strike_price = cols[5].text
            put_price = cols[9].text
            put_oi = cols[10].text

            calls.append({"OI": call_oi, "Price": call_price, "Strike": strike_price})
            puts.append({"Strike": strike_price, "Price": put_price, "OI": put_oi})

        driver.quit()
        return pd.DataFrame(calls), pd.DataFrame(puts)

    except Exception as e:
        driver.quit()
        return None, None

# ---------------------------
# 3. ML Model Training
# ---------------------------
def train_lstm(X_train, y_train):
    """Train LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# ---------------------------
# 4. Prediction & Analysis
# ---------------------------
def ensemble_predict(asset_data):
    """Combine predictions from multiple models"""
    predictions = {}
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(asset_data['Close'].values.reshape(-1, 1))

    if len(scaled_data) > 60:
        X = np.array([scaled_data[i-60:i, 0] for i in range(60, len(scaled_data))])
        y = scaled_data[60:]

        lstm_model = train_lstm(X.reshape(X.shape[0], X.shape[1], 1), y)
        xgb_model = train_xgboost(X, y)
        rf_model = train_random_forest(X, y)

        last_sequence = X[-1].reshape(1, 60, 1)
        lstm_pred = scaler.inverse_transform(lstm_model.predict(last_sequence))[0][0]
        xgb_pred = scaler.inverse_transform(xgb_model.predict(X[-1].reshape(1, -1)))[0]
        rf_pred = scaler.inverse_transform(rf_model.predict(X[-1].reshape(1, -1)))[0]

        predictions['ensemble'] = (0.4 * lstm_pred) + (0.3 * xgb_pred) + (0.3 * rf_pred)

    return predictions

# ---------------------------
# 5. Streamlit Dashboard
# ---------------------------
def main():
    st.title("Multi-Asset Advanced Analyzer")

    assets = st.sidebar.text_input("Enter assets (comma-separated)", "^NSEI, ^NSEBANK, RELIANCE.NS").split(',')
    all_data = fetch_live_data([a.strip() for a in assets])
    vix = fetch_vix()

    for asset in [a.strip() for a in assets]:
        st.header(f"{asset} Analysis")

        try:
            asset_data = all_data[asset] if len(assets) > 1 else all_data
            analysis = ensemble_predict(asset_data)

            col1, col2 = st.columns(2)
            col1.metric("Current Price", f"₹{asset_data['Close'].iloc[-1]:.2f}")
            col2.metric("VIX", f"{vix:.2f}" if asset == "^NSEI" else "N/A")

            if 'ensemble' in analysis:
                st.subheader("Ensemble Prediction")
                st.write(f"Next 15-min prediction: ₹{analysis['ensemble']:.2f}")

            if asset in ["^NSEI", "^NSEBANK"]:
                st.subheader("Option Chain Analysis")
                calls, puts = fetch_option_chain(asset.replace("^", ""))
                if calls is not None and puts is not None:
                    st.write("Top 5 Calls:")
                    st.dataframe(calls[['Strike', 'OI', 'Price']].head())
                    st.write("Top 5 Puts:")
                    st.dataframe(puts[['Strike', 'OI', 'Price']].head())
                else:
                    st.write("Option chain data not available.")

            st.subheader("Price Chart")
            fig, ax = plt.subplots()
            ax.plot(asset_data['Close'], label='Price')
            ax.set_title(f"{asset} Price Movement")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error analyzing {asset}: {str(e)}")

if __name__ == "__main__":
    main()
