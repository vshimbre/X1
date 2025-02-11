import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from nsepython import nse_optionchain_scrapper
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------
# 1. Multi-Asset Data Fetching
# ---------------------------
def fetch_live_data(symbols=["^NSEI", "^NSEBANK"], interval="15m"):
    """Fetch data for multiple assets"""
    try:
        data = yf.download(tickers=symbols, period="1d", interval=interval, group_by='ticker')
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def fetch_vix():
    """Fetch India VIX data"""
    try:
        vix = yf.download("^INDIAVIX", period="1d", interval="15m")
        return float(vix['Close'].iloc[-1])  # Convert to float for formatting
    except Exception:
        return None

def fetch_option_chain(index="NIFTY"):
    """Fetch option chain for given index"""
    try:
        oc_data = nse_optionchain_scrapper(index)
        calls = pd.DataFrame(oc_data['CE']['data'])
        puts = pd.DataFrame(oc_data['PE']['data'])
        return calls, puts
    except Exception:
        return None, None

# ---------------------------
# 2. Advanced ML Models
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

def train_prophet(data):
    """Train Prophet model"""
    model = Prophet(daily_seasonality=True)
    df = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model.fit(df)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# ---------------------------
# 3. Ensemble Predictions
# ---------------------------
def ensemble_predict(asset_data):
    """Combine predictions from multiple models"""
    predictions = {}
    
    try:
        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(asset_data['Close'].values.reshape(-1,1))
    
        if len(scaled_data) > 60:
            X = np.array([scaled_data[i-60:i, 0] for i in range(60, len(scaled_data))])
            y = scaled_data[60:]
            
            # Train models
            lstm_model = train_lstm(X.reshape(X.shape[0], X.shape[1], 1), y)
            xgb_model = train_xgboost(X, y)
            rf_model = train_random_forest(X, y)
            
            # Make predictions
            last_sequence = X[-1].reshape(1, 60, 1)
            lstm_pred = scaler.inverse_transform(lstm_model.predict(last_sequence))[0][0]
            xgb_pred = scaler.inverse_transform(xgb_model.predict(X[-1].reshape(1, -1)))[0]
            rf_pred = scaler.inverse_transform(rf_model.predict(X[-1].reshape(1, -1)))[0]
            
            # Weighted average of predictions
            predictions['ensemble'] = (0.4 * lstm_pred) + (0.3 * xgb_pred) + (0.3 * rf_pred)
    
    except Exception as e:
        st.error(f"Error in model prediction: {e}")

    return predictions

# ---------------------------
# 4. Streamlit Dashboard
# ---------------------------
def main():
    st.title("Multi-Asset Advanced Analyzer")

    # User input for assets
    assets = st.sidebar.text_input("Enter assets (comma-separated)", "^NSEI, ^NSEBANK, RELIANCE.NS").split(',')
    
    # Fetch data for all assets
    all_data = fetch_live_data([a.strip() for a in assets])
    vix = fetch_vix()
    
    if all_data is None:
        st.error("No data available.")
        return

    # Analyze each asset
    for asset in [a.strip() for a in assets]:
        st.header(f"{asset} Analysis")

        try:
            # Get asset data
            asset_data = all_data[asset] if len(assets) > 1 else all_data
            
            # Ensure asset data is valid
            if asset_data is None or asset_data.empty:
                st.warning(f"No data for {asset}. Skipping...")
                continue

            # Perform technical analysis
            analysis = ensemble_predict(asset_data)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            current_price = float(asset_data['Close'].iloc[-1])  # Convert Series to float
            col1.metric("Current Price", f"₹{current_price:.2f}")

            # Show VIX only for NIFTY
            if asset == "^NSEI" and vix is not None:
                col3.metric("VIX", f"{vix:.2f}")
            else:
                col3.metric("VIX", "N/A")
            
            # Show predictions
            if 'ensemble' in analysis:
                st.subheader("Ensemble Prediction")
                st.write(f"Next 15-min prediction: ₹{analysis['ensemble']:.2f}")
            
            # Option chain for indices
            if asset in ["^NSEI", "^NSEBANK"]:
                st.subheader("Option Chain Analysis")
                calls, puts = fetch_option_chain(asset.replace("^", ""))
                
                if calls is not None and puts is not None:
                    st.write("Top 5 Calls:")
                    st.dataframe(calls[['strikePrice', 'openInterest', 'lastPrice']].head())
                    st.write("Top 5 Puts:")
                    st.dataframe(puts[['strikePrice', 'openInterest', 'lastPrice']].head())
                else:
                    st.warning("Option chain data not available.")
            
            # Visualization
            st.subheader("Price Chart")
            fig, ax = plt.subplots()
            ax.plot(asset_data['Close'], label='Price', color='blue')
            ax.set_title(f"{asset} Price Movement")
            ax.legend()
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error analyzing {asset}: {str(e)}")

if __name__ == "__main__":
    main()
