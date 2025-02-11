import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------
# 1. Data Fetching
# ---------------------------
def fetch_live_data(symbols=["^NSEI", "^NSEBANK"], interval="15m"):
    """Fetch stock/index data"""
    try:
        data = yf.download(tickers=symbols, period="1d", interval=interval, group_by='ticker')
        return data
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return None

def fetch_vix():
    """Fetch India VIX"""
    try:
        vix = yf.download("^INDIAVIX", period="1d", interval="15m")
        return round(vix['Close'].iloc[-1], 2)
    except:
        return "Unavailable"

def fetch_option_chain(index="NIFTY"):
    """Try fetching NSE option chain via alternative method"""
    try:
        import nsetools
        nse = nsetools.Nse()
        options = nse.get_option_chain(index)
        if options:
            calls = pd.DataFrame(options['CE'])
            puts = pd.DataFrame(options['PE'])
            return calls, puts
    except:
        return None, None

# ---------------------------
# 2. Streamlit UI
# ---------------------------
def main():
    st.title("NIFTY & BankNIFTY Analysis")

    asset = st.sidebar.selectbox("Select Asset", ["^NSEI", "^NSEBANK"])
    
    # Fetch data
    asset_data = fetch_live_data([asset])
    vix = fetch_vix()
    
    if asset_data is not None and not asset_data.empty:
        st.subheader(f"{asset} Live Data")
        price = asset_data['Close'].iloc[-1]
        
        # **🔹 FIXED: Proper Formatting**
        col1, col2 = st.columns(2)
        col1.metric("Current Price", f"₹{price:.2f}")
        col2.metric("VIX", vix)
        
        # Option Chain
        st.subheader("Option Chain Analysis")
        calls, puts = fetch_option_chain(asset.replace("^", ""))
        if calls is not None and puts is not None:
            st.write("Calls Data (Top 5):")
            st.dataframe(calls.head())
            st.write("Puts Data (Top 5):")
            st.dataframe(puts.head())
        else:
            st.warning("Option chain data unavailable.")

        # Chart
        st.subheader("Price Chart")
        fig, ax = plt.subplots()
        ax.plot(asset_data.index, asset_data['Close'], color='blue')
        ax.set_title(f"{asset} Price Movement")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No data available.")

if __name__ == "__main__":
    main()
