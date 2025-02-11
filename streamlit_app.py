import os
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# ---------------------------
# 1. Setup Selenium Driver for Streamlit Cloud
# ---------------------------
@st.cache_resource
def get_selenium_driver():
    chrome_bin = "/usr/bin/google-chrome"
    chromedriver_bin = "/usr/bin/chromedriver"

    if not os.path.exists(chrome_bin) or not os.path.exists(chromedriver_bin):
        st.info("Installing Chrome & Chromedriver...")
        subprocess.run("apt-get update", shell=True)
        subprocess.run("apt-get install -y chromium-browser chromium-chromedriver", shell=True)

    chrome_options = Options()
    chrome_options.binary_location = chrome_bin
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(chromedriver_bin)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# ---------------------------
# 2. Fetch NIFTY 50 Stock Data
# ---------------------------
def get_nifty_data():
    st.subheader("📈 NIFTY 50 Stock Market Analysis")
    ticker = "^NSEI"
    data = yf.download(ticker, period="1mo", interval="1d")

    if data.empty:
        st.error("Error fetching NIFTY data.")
        return

    st.write(data.tail())

    # Plot price chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["Close"], label="NIFTY 50 Closing Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# 3. Scrape Option Chain Data
# ---------------------------
def get_option_chain():
    st.subheader("📜 NIFTY 50 Option Chain Data")
    driver = get_selenium_driver()

    url = "https://www.nseindia.com/option-chain"
    driver.get(url)
    driver.implicitly_wait(10)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    try:
        table = soup.find("table", {"id": "optionChainTable"})
        df = pd.read_html(str(table))[0]

        # Display table
        st.dataframe(df)
    except Exception as e:
        st.error(f"Option Chain Data Not Available: {e}")

    driver.quit()

# ---------------------------
# 4. Scrape India VIX Data
# ---------------------------
def get_vix():
    st.subheader("⚡ India VIX Data")
    driver = get_selenium_driver()

    url = "https://www.nseindia.com/market-data/india-vix"
    driver.get(url)
    driver.implicitly_wait(10)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    try:
        vix_value = soup.find("div", class_="vix-value").text.strip()
        st.success(f"📊 Current India VIX: {vix_value}")
    except Exception as e:
        st.error(f"VIX Data Not Available: {e}")

    driver.quit()

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("📊 NIFTY 50 Stock Market Analysis App")
st.sidebar.title("🔍 Features")

menu = st.sidebar.radio("Choose an option:", ["NIFTY 50 Data", "Option Chain", "India VIX"])

if menu == "NIFTY 50 Data":
    get_nifty_data()
elif menu == "Option Chain":
    get_option_chain()
elif menu == "India VIX":
    get_vix()
