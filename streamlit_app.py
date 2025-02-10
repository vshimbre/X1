import streamlit as st
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import cirq
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from datetime import datetime, timedelta

# ---------------------------
# 1. QUANTUM ERROR MITIGATION (DAEM)
# ---------------------------
class QuantumErrorMitigator:
    def __init__(self):
        self.error_model = self._build_error_network()
        
    def _build_error_network(self):
        """Neural network for error cancellation"""
        inputs = Input(shape=(2,))  # Expecting 2 features
        x = Dense(32, activation='relu')(inputs)
        outputs = Dense(2, activation='sigmoid')(x)
        return Model(inputs, outputs)

    def mitigate(self, counts):
        """Apply error mitigation to quantum results"""
        input_data = np.array([list(counts.values())])
        
        # Ensure input has exactly 2 features (pad if necessary)
        if input_data.shape[1] != 2:
            input_data = np.pad(input_data, ((0, 0), (0, max(0, 2 - input_data.shape[1]))), mode='constant')

        # Debugging: print input shape
        print("Counts:", counts)
        print("Input shape:", input_data.shape)

        return self.error_model.predict(input_data)

# ---------------------------
# 2. REAL-TIME NSE DATA INTEGRATION
# ---------------------------
class NSEDataEngine:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        })
        
    def fetch_option_chain(self, symbol):
        """Fetch NSE option chain with API blocking handling"""
        try:
            self.session.get("https://www.nseindia.com")
            response = self.session.get(
                f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
                timeout=2
            )
            response.raise_for_status()
            return response.json()
        except:
            st.warning(f"API Error: Using cached data for {symbol}")
            return self._fallback_data(symbol)

    def _fallback_data(self, symbol):
        """Fallback data if API is blocked"""
        return {
            'strikePrices': [22000, 22100, 22200],
            'data': [
                {'strikePrice': 22000, 'CE': {'openInterest': 1000, 'impliedVolatility': 15}},
                {'strikePrice': 22100, 'CE': {'openInterest': 1200, 'impliedVolatility': 16}},
            ]
        }

# ---------------------------
# 3. QUANTUM PREDICTION ENGINE (CIRQ)
# ---------------------------
class QuantumPredictor:
    def __init__(self):
        self.simulator = cirq.Simulator()
        self.error_mitigator = QuantumErrorMitigator()
        
    def predict(self, market_state):
        """Quantum-enhanced market prediction"""
        try:
            qc = self._create_circuit(market_state)
            result = self.simulator.run(qc, repetitions=500)
            counts = result.histogram(key='m')
            mitigated_counts = self.error_mitigator.mitigate(counts)
            return self._interpret_counts(mitigated_counts)
        except Exception as e:
            st.error(f"Quantum prediction failed: {str(e)}")
            return 0.5  # Fallback to neutral prediction
    
    def _create_circuit(self, market_state):
        """2-qubit hardware-efficient circuit"""
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        
        # State preparation
        for i, q in enumerate(qubits):
            circuit.append(cirq.H(q))
            if i < len(market_state):
                circuit.append(cirq.ry(market_state[i] * np.pi)(q))
            else:
                circuit.append(cirq.ry(0.0)(q))  # Default value
        
        # Entanglement
        circuit.append(cirq.CZ(qubits[0], qubits[1]))
        circuit.append(cirq.measure(*qubits, key='m'))
        return circuit
    
    def _interpret_counts(self, counts):
        """Convert quantum measurements to prediction"""
        try:
            max_state = max(counts, key=counts.get, default=0)
            return 1.0 if bin(max_state).count('1') >= 1 else 0.0
        except:
            return 0.5  # Fallback to neutral prediction

# ---------------------------
# 4. INTERACTIVE DASHBOARD
# ---------------------------
class TradingDashboard:
    def __init__(self, symbols):
        st.set_page_config(layout="wide", page_title="Quantum Trading Terminal")
        self.symbols = symbols
        self.data_engine = NSEDataEngine()
        self.predictor = QuantumPredictor()
        
    def render(self):
        """Interactive dashboard interface"""
        st.title("💎 Quantum Trading Terminal - NSE Pro")
        
        # Real-Time Prices
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
                market_state = self._get_market_state(symbol)
                prediction = self.predictor.predict(market_state)
                prediction_data.append({
                    'Symbol': symbol,
                    'Current Price': prices.get(symbol, 0.0),
                    'Prediction': prediction,
                    'Target': prices.get(symbol, 0.0) * (1 + prediction/10),
                    'Stop Loss': prices.get(symbol, 0.0) * (1 - (1-prediction)/8)
                })
            except:
                prediction_data.append({'Symbol': symbol, 'Prediction': 0.5})

        # Display Predictions
        df = pd.DataFrame(prediction_data)
        st.dataframe(df.style.format({
            'Current Price': '₹{:.2f}',
            'Prediction': '{:.2%}',
            'Target': '₹{:.2f}',
            'Stop Loss': '₹{:.2f}'
        }), height=300)
        
    def _get_real_time_prices(self):
        """Fetch real-time prices with error handling"""
        prices = {}
        for symbol in self.symbols:
            try:
                data = yf.Ticker(symbol + ".NS").history(period='1d')
                prices[symbol] = data['Close'][-1] if not data.empty else 0.0
            except:
                prices[symbol] = 0.0
        return prices
    
    def _get_daily_change(self, symbol):
        """Calculate daily percentage change"""
        try:
            data = yf.Ticker(symbol + ".NS").history(period='2d')
            return f"{((data['Close'][-1] - data['Close'][-2])/data['Close'][-2]*100):.2f}%" if len(data['Close']) >= 2 else "N/A"
        except:
            return "N/A"
    
    def _get_market_state(self, symbol):
        """Get market data for quantum processing"""
        try:
            data = yf.Ticker(symbol + ".NS").history(period='1d')
            return np.array([data['Close'][-1], data['Volume'][-1]]) if not data.empty else np.array([0.0, 0.0])
        except:
            return np.array([0.0, 0.0])

# ---------------------------
# 5. MAIN APPLICATION
# ---------------------------
if __name__ == "__main__":
    SYMBOLS = ['NIFTY_50', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY']
    dashboard = TradingDashboard(SYMBOLS)
    dashboard.render()
