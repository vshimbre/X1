import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import cirq
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# ---------------------------
# 1. QUANTUM ERROR MITIGATION (DAEM)
# ---------------------------
class QuantumErrorMitigator:
    def __init__(self):
        self.error_model = self._build_error_network()

    def _build_error_network(self):
        """Lightweight neural network for error cancellation"""
        inputs = Input(shape=(2,))
        x = Dense(16, activation='relu')(inputs)
        outputs = Dense(2, activation='sigmoid')(x)
        return Model(inputs, outputs)

    def mitigate(self, counts):
        """Apply error mitigation to quantum results"""
        return self.error_model.predict(np.array([list(counts.values())]))

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
        """2-qubit circuit for quantum analysis"""
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
# 3. INTERACTIVE DASHBOARD
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
                market_state = self._get_market_state(symbol)
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
        """Fetch real-time prices with correct index mapping"""
        prices = {}
        symbol_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}

        for symbol in self.symbols:
            yf_symbol = symbol_map.get(symbol, symbol + ".NS")  # Map indexes

            try:
                data = yf.Ticker(yf_symbol).history(period='1d')
                if not data.empty and "Close" in data.columns:
                    prices[symbol] = data["Close"].iloc[-1]
                else:
                    prices[symbol] = 0.0
                    st.warning(f"No data found for {symbol}")
            except Exception as e:
                st.error(f"Failed to fetch price for {symbol}: {str(e)}")
                prices[symbol] = 0.0

        return prices

    def _get_daily_change(self, symbol):
        """Calculate daily price change"""
        try:
            symbol_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
            yf_symbol = symbol_map.get(symbol, symbol + ".NS")

            data = yf.Ticker(yf_symbol).history(period='2d')
            if len(data) >= 2:
                return f"{((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%"
            return "N/A"
        except Exception as e:
            st.error(f"Daily change failed for {symbol}: {str(e)}")
            return "N/A"

    def _get_market_state(self, symbol):
        """Fetch market data for quantum processing"""
        try:
            symbol_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
            yf_symbol = symbol_map.get(symbol, symbol + ".NS")

            data = yf.Ticker(yf_symbol).history(period='1d')
            if len(data) > 0:
                return np.array([data['Close'].iloc[-1], data['Volume'].iloc[-1]])
            return np.array([0.0, 0.0])
        except Exception as e:
            st.error(f"Market state fetch failed for {symbol}: {str(e)}")
            return np.array([0.0, 0.0])

# ---------------------------
# 4. MAIN APPLICATION
# ---------------------------
if __name__ == "__main__":
    SYMBOLS = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
    dashboard = TradingDashboard(SYMBOLS)
    dashboard.render()
