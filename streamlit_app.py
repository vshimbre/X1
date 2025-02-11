import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit Title
st.title("NIFTY Stock Market Analysis & Prediction")

# Function to fetch stock data
def fetch_stock_data(ticker="^NSEI", period="7d", interval="15m"):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Train LSTM model
def train_lstm(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    return model

# Train XGBoost model
def train_xgboost(X, y):
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50)
    model.fit(X, y)
    return model

# Train Random Forest model
def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

# Prediction function
def ensemble_predict(asset_data):
    """Predict next 15-min movement with target price."""
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

        # Weighted Ensemble Prediction
        predicted_price = (0.4 * lstm_pred) + (0.3 * xgb_pred) + (0.3 * rf_pred)
        current_price = asset_data['Close'].iloc[-1]

        # Determine Direction and Target
        direction = "Up" if predicted_price > current_price else "Down"
        target_price = predicted_price + (0.002 * current_price) if direction == "Up" else predicted_price - (0.002 * current_price)

        predictions['ensemble'] = predicted_price
        predictions['direction'] = direction
        predictions['target'] = target_price

    return predictions

# Function to scrape NSE option chain data
def fetch_option_chain():
    url = "https://www.nseindia.com/option-chain"
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        driver.get(url)
        driver.implicitly_wait(10)
        
        # Extract data table
        table = driver.find_element(By.CLASS_NAME, "opttbldata")  # Update class as per NSE site
        rows = table.find_elements(By.TAG_NAME, "tr")

        option_data = []
        for row in rows[1:]:  # Skip header
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) > 0:
                option_data.append([col.text for col in cols])

        driver.quit()
        return pd.DataFrame(option_data, columns=["Calls OI", "Calls Change OI", "Calls Volume", "Calls IV",
                                                  "Calls LTP", "Calls Net Chng", "Calls Bid Price", "Calls Ask Price",
                                                  "Strike Price",
                                                  "Puts Bid Price", "Puts Ask Price", "Puts Net Chng", "Puts LTP",
                                                  "Puts IV", "Puts Volume", "Puts Change OI", "Puts OI"])
    except Exception as e:
        driver.quit()
        st.error(f"Error fetching option chain data: {e}")
        return None

# Fetch stock data
df = fetch_stock_data()

if df is not None:
    st.subheader("Price Chart")
    st.line_chart(df['Close'])

    # Run predictions
    analysis = ensemble_predict(df)

    if 'ensemble' in analysis:
        st.subheader("Next 15-min Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Price", f"₹{analysis['ensemble']:.2f}")

        # Show Up/Down movement
        direction_color = "green" if analysis['direction'] == "Up" else "red"
        col2.markdown(f"<h3 style='color:{direction_color};'>⬆ {analysis['direction']}</h3>", unsafe_allow_html=True)

        # Target Price
        col3.metric("Target Price", f"₹{analysis['target']:.2f}")

    # Fetch option chain data
    option_chain_df = fetch_option_chain()
    if option_chain_df is not None:
        st.subheader("NIFTY Option Chain")
        st.dataframe(option_chain_df)
