import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import cirq
import requests
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# ---------------------------
# 1. QUANTUM ERROR MITIGATION (DAEM)
# class QuantumErrorMitigator:
    def __init__(self):
        self.error_model = self._build_error_network()

    def _build_error_network(self):
        """Lightweight neural network for error cancellation"""
        inputs = Input(shape=(2,))  # Expecting a 2-feature input
        x = Dense(16, activation='relu')(inputs)
        outputs = Dense(2, activation='sigmoid')(x)
        return Model(inputs, outputs)

    def mitigate(self, counts):
        """Apply error mitigation to quantum results"""
        values = list(counts.values())

        # Ensure exactly 2 values are passed (pad/truncate if necessary)
        if len(values) < 2:
            values += [0] * (2 - len(values))  # Pad with zeros if less than 2
        elif len(values) > 2:
            values = values[:2]  # Truncate to 2 values
        
        input_array = np.array([values])  # Convert to proper shape (1, 2)
        return self.error_model.predict(input_array)
# ---------------------------
# 2. QUANTUM PREDICTION ENGINE (CIRQ)
# ---------------------------
class QuantumPredictor:
    def __init__(self):
        self.simulator = cirq.Simulator()
        self.error_mitigator = QuantumErrorMitigator()

    def predict(self, market_state):
        """Quantum-enhanced market prediction"""
        try:
            qc = self._create_circuit(market_state)
            result = self.simulator.run(qc, repetitions=250)
            counts = result.histogram(key='m')
            mitigated_counts = self.error_mitigator.mitigate(counts)
            return self._interpret_counts(mitigated_counts)
        except Exception as e:
            st.error(f"Quantum prediction failed: {str(e)}")
            return 0.5  # Fallback to neutral prediction

    def _create_circuit(self, market_state):
        """2-qubit quantum circuit for prediction"""
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()

        # State preparation
        for i, q in enumerate(qubits):
            circuit.append(cirq.H(q))
            circuit.append(cirq.ry(market_state[i] * np.pi)(q))

        # Entanglement & measurement
        circuit.append(cirq.CZ(qubits[0], qubits[1]))
        circuit.append(cirq.measure(*qubits, key='m'))
        return circuit

    def _interpret_counts(self, counts):
        """Convert quantum results into a market prediction"""
        try:
            max_state = max(counts, key=counts.get)
            return 1.0 if bin(max_state).count('1') >= 1 else 0.0
        except:
            return 0.5  # Fallback to neutral prediction

# ---------------------------
# 3. NSE DATA FETCHING (NIFTY & BANKNIFTY)
# ---------------------------
def get_nse_index_price(index_name):
    """Fetch real-time price of NIFTY and BANKNIFTY from NSE India"""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={index_name}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)  # Bypass bot protection
    response = session.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if "records" in data and "underlyingValue" in data["records"]:
            return data["records"]["underlyingValue"]

    return None  # If data fetching fails

# ---------------------------
# 4. INTERACTIVE DASHBOARD
# ---------------------------
class TradingDashboard:
    def __init__(self, symbols):
        st.set_page_config(layout="wide", page_title="Quantum Trading Terminal")
        self.symbols = symbols
        self.predictor = QuantumPredictor()

    def render(self):
        """Interactive dashboard"""
        st.title("💎 Quantum Trading Terminal - Mobile")

        # Fetch real-time prices
        prices = self._get_real_time_prices()
        cols = st.columns(len(self.symbols))

        for idx, symbol in enumerate(self.symbols):
            cols[idx].metric(
                f"{symbol} Price", 
                f"₹{prices.get(symbol, 0.0):.2f}",
                self._get_daily_change(symbol)
            )

        # Quantum Predictions
        st.header("Quantum Predictions")
        prediction_data = []
        for symbol in self.symbols:
            try:
                market_state = self._get_market_state(symbol, prices)
                prediction = self.predictor.predict(market_state)
                prediction_data.append({
                    'Symbol': symbol,
                    'Current Price': prices.get(symbol, 0.0),
                    'Prediction': prediction,
                    'Target': prices.get(symbol, 0.0) * (1 + prediction/10),
                    'Stop Loss': prices.get(symbol, 0.0) * (1 - (1-prediction)/8)
                })
            except Exception as e:
                st.error(f"Prediction failed for {symbol}: {str(e)}")
                prediction_data.append({
                    'Symbol': symbol,
                    'Current Price': 0.0,
                    'Prediction': 0.5,
                    'Target': 0.0,
                    'Stop Loss': 0.0
                })

        # Display Predictions
        df = pd.DataFrame(prediction_data)
        st.dataframe(
            df.style.format({
                'Current Price': '₹{:.2f}',
                'Prediction': '{:.2%}',
                'Target': '₹{:.2f}',
                'Stop Loss': '₹{:.2f}'
            }),
            height=300
        )

    def _get_real_time_prices(self):
        """Fetch real-time prices from NSE for indices and Yahoo Finance for stocks"""
        prices = {}

        # Fetch NIFTY & BANKNIFTY from NSE
        for index in ["NIFTY", "BANKNIFTY"]:
            try:
                price = get_nse_index_price(index)
                if price:
                    prices[index] = price
                else:
                    st.warning(f"NSE data unavailable for {index}")
            except Exception as e:
                st.error(f"Failed to fetch {index} from NSE: {str(e)}")
                prices[index] = 0.0

        # Fetch stocks from Yahoo Finance
        for symbol in self.symbols:
            if symbol in ["NIFTY", "BANKNIFTY"]:
                continue  # Skip, already fetched from NSE

            try:
                data = yf.Ticker(symbol + ".NS").history(period="1d")
                if not data.empty:
                    prices[symbol] = data["Close"].iloc[-1]
                else:
                    st.warning(f"No Yahoo data for {symbol}")
                    prices[symbol] = 0.0
            except Exception as e:
                st.error(f"Failed to fetch {symbol} from Yahoo: {str(e)}")
                prices[symbol] = 0.0

        return prices

    def _get_daily_change(self, symbol):
        """Calculate daily price change"""
        try:
            data = yf.Ticker(symbol + ".NS").history(period='2d')
            if len(data) >= 2:
                return f"{((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%"
            return "N/A"
        except Exception as e:
            st.error(f"Daily change failed for {symbol}: {str(e)}")
            return "N/A"

    def _get_market_state(self, symbol, prices):
        """Generate market state for quantum processing"""
        if symbol in ["NIFTY", "BANKNIFTY"]:
            return np.array([prices.get(symbol, 0.0), 0])  # NSE indices (No volume data)
        
        try:
            data = yf.Ticker(symbol + ".NS").history(period="1d")
            if not data.empty:
                return np.array([data["Close"].iloc[-1], data["Volume"].iloc[-1]])
        except:
            pass
        
        return np.array([0.0, 0.0])  # Default fallback

# ---------------------------
# 5. MAIN APPLICATION
# ---------------------------
if __name__ == "__main__":
    SYMBOLS = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
    dashboard = TradingDashboard(SYMBOLS)
    dashboard.render()
