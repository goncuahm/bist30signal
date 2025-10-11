import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="BIST30 RSI Strategy & Forecast", layout="wide")

# ------------------------------
# Title
# ------------------------------
st.title("📊 BIST30 RSI Strategy — Backtest & LSTM Forecast")

# ------------------------------
# Sidebar Parameters
# ------------------------------
st.sidebar.header("🔧 Strategy Parameters")

period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "3y"], index=1)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 9)
buy_threshold = st.sidebar.slider("Buy Threshold (RSI < x1)", 5, 45, 40)
sell_threshold = st.sidebar.slider("Sell Threshold (RSI > x2)", 55, 95, 63)
tcost = st.sidebar.number_input("Transaction Cost (e.g., 0.002 = 0.2%)", value=0.002, step=0.0005)

# ------------------------------
# Define BIST30 tickers
# ------------------------------
bist30_tickers = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS", "EKGYO.IS", "ENJSA.IS", "EREGL.IS",
    "FROTO.IS", "GARAN.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS", "KOZAL.IS",
    "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
    "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "TTRAK.IS", "HALKB.IS", "ALARK.IS"
]

# ------------------------------
# RSI Function
# ------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ------------------------------
# Backtest Function
# ------------------------------
def backtest_strategy(df, x1, x2, tcost):
    open_positions = []
    closed_trades = []
    for i in range(1, len(df)):
        rsi = df["RSI"].iloc[i]
        price = df["Close"].iloc[i]
        date = df.index[i]

        if rsi < x1:
            open_positions.append({"entry_price": price, "entry_date": date})
        elif rsi > x2 and open_positions:
            entry = open_positions.pop(0)
            closed_trades.append({
                "buy_date": entry["entry_date"],
                "buy_price": entry["entry_price"],
                "sell_date": date,
                "sell_price": price,
                "return": (price - entry["entry_price"]) / entry["entry_price"] - tcost
            })

    total_return = np.sum([t["return"] for t in closed_trades])
    avg_return = np.mean([t["return"] for t in closed_trades]) if closed_trades else 0
    return total_return, avg_return, closed_trades

# ------------------------------
# Analysis Loop
# ------------------------------
st.subheader("🔍 Scanning BIST30 Stocks...")

results = []
buy_signals = []
sell_signals = []

for ticker in bist30_tickers:
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty:
            continue
        data["RSI"] = compute_rsi(data["Close"], rsi_period)
        data = data.dropna()

        total_return, avg_return, trades = backtest_strategy(data, buy_threshold, sell_threshold, tcost)
        latest_rsi = float(data["RSI"].iloc[-1])
        latest_close = float(data["Close"].iloc[-1])

        if latest_rsi < buy_threshold:
            signal = "BUY"
            buy_signals.append((ticker, latest_close, latest_rsi))
        elif latest_rsi > sell_threshold:
            signal = "SELL"
            sell_signals.append((ticker, latest_close, latest_rsi))
        else:
            signal = "HOLD"

        results.append({
            "Ticker": ticker,
            "Signal": signal,
            "Latest RSI": round(latest_rsi, 2),
            "Latest Close": round(latest_close, 2),
            "Cumulative Return (%)": round(total_return * 100, 2),
            "Return per Trade (%)": round(avg_return * 100, 2),
            "Number of Trades": len(trades)
        })

    except Exception as e:
        print(f"Error with {ticker}: {e}")

# ------------------------------
# Convert to DataFrame
# ------------------------------
results_df = pd.DataFrame(results).sort_values(by="Return per Trade (%)", ascending=False)
buy_df = pd.DataFrame(buy_signals, columns=["Ticker", "Close Price", "RSI"])
sell_df = pd.DataFrame(sell_signals, columns=["Ticker", "Close Price", "RSI"])

# ------------------------------
# Display Results
# ------------------------------
st.subheader("📈 RSI Strategy Results (BIST30)")
st.dataframe(results_df, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🟢 Current BUY Signals")
    st.dataframe(buy_df if not buy_df.empty else pd.DataFrame([["-", "-", "-"]], columns=["Ticker","Close Price","RSI"]))

with col2:
    st.subheader("🔴 Current SELL Signals")
    st.dataframe(sell_df if not sell_df.empty else pd.DataFrame([["-", "-", "-"]], columns=["Ticker","Close Price","RSI"]))

# ================================================================
# PART 2: Select Stock for LSTM Forecast
# ================================================================
st.subheader("🤖 LSTM RSI Forecast (User-Selected Stock)")

selected_ticker = st.selectbox("Select a stock for RSI forecast:", ["None"] + bist30_tickers)

# ------------------------------
# LSTM Function
# ------------------------------
def lstm_forecast_rsi(rsi_series, n_past=9, n_future=4):
    if len(rsi_series) < n_past + 5:
        return [np.nan] * n_future

    scaler = MinMaxScaler(feature_range=(0, 1))
    rsi_scaled = scaler.fit_transform(rsi_series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(n_past, len(rsi_scaled) - n_future):
        X.append(rsi_scaled[i - n_past:i, 0])
        y.append(rsi_scaled[i:i + n_future, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_past, 1)),
        Dense(25, activation='relu'),
        Dense(n_future)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=8, verbose=0)

    last_window = rsi_scaled[-n_past:].reshape((1, n_past, 1))
    forecast_scaled = model.predict(last_window)
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    return forecast

# ------------------------------
# Run forecast only if user selected a stock
# ------------------------------
if selected_ticker != "None":
    st.write(f"### 🔮 Forecasting RSI for: **{selected_ticker}**")
    data = yf.download(selected_ticker, period=period, auto_adjust=True, progress=False)
    data["RSI"] = compute_rsi(data["Close"], rsi_period)
    data = data.dropna()

    forecast = lstm_forecast_rsi(data["RSI"], n_past=9, n_future=4)

    # Show forecast table
    forecast_df = pd.DataFrame({
        "Ticker": [selected_ticker],
        "Day+1 RSI": [round(forecast[0], 2)],
        "Day+2 RSI": [round(forecast[1], 2)],
        "Day+3 RSI": [round(forecast[2], 2)],
        "Day+4 RSI": [round(forecast[3], 2)],
    })
    st.dataframe(forecast_df, use_container_width=True)

    # Mini plot for Close price
    st.write("📉 Recent Close Price Trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index[-100:], data["Close"].iloc[-100:], label="Close", color="steelblue")
    ax.set_title(f"{selected_ticker} — Recent Close Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Select a stock above to generate RSI LSTM forecast.")

st.caption("Developed for educational and research purposes — RSI Strategy + LSTM Forecast on BIST30.")
