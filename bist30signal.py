import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="BIST30 Technical & Fundamental Strategy with Machine Forecast", layout="wide")

# ------------------------------
# Title
# ------------------------------
st.title("📊 BIST30 Technical Strategy — Backtest & LSTM Forecast")

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
# EPS Function
# ------------------------------
def get_eps(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        eps = info.get('trailingEps', None)
        return eps if eps is not None else np.nan
    except:
        return np.nan

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

# Store RSI data for RSI grid plot
rsi_store = {}    # ticker -> {"rsi": Series, "signal": str}

# Store price/MA data for price grid plot
price_store = {}  # ticker -> {"close": Series, "ma50": Series, "ma200": Series, "signal": str}

for ticker in bist30_tickers:
    try:
        # Fetch 2y of data so MA200 has enough history; display window = last 252 days
        data = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if data.empty:
            continue

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data["RSI"] = compute_rsi(data["Close"], rsi_period)
        data = data.dropna()

        total_return, avg_return, trades = backtest_strategy(data, buy_threshold, sell_threshold, tcost)
        latest_rsi = float(data["RSI"].iloc[-1])
        latest_close = float(data["Close"].iloc[-1])

        eps = get_eps(ticker)

        if latest_rsi < buy_threshold:
            if not np.isnan(eps) and eps > 0:
                signal = "BUY"
                buy_signals.append((ticker, latest_close, latest_rsi, eps))
            else:
                signal = "HOLD"
        elif latest_rsi > sell_threshold:
            signal = "SELL"
            sell_signals.append((ticker, latest_close, latest_rsi, eps))
        else:
            signal = "HOLD"

        # RSI grid — last 252 rows
        data_252 = data.iloc[-252:]
        rsi_store[ticker] = {"rsi": data_252["RSI"].copy(), "signal": signal}

        # Price/MA grid — compute MAs on full dataset, then slice last 252 days
        close_full = data["Close"]
        ma50_full  = close_full.rolling(window=50).mean()
        ma200_full = close_full.rolling(window=200).mean()
        price_store[ticker] = {
            "close":  close_full.iloc[-252:],
            "ma50":   ma50_full.iloc[-252:],
            "ma200":  ma200_full.iloc[-252:],
            "signal": signal,
        }

        results.append({
            "Ticker": ticker,
            "Signal": signal,
            "Latest RSI": round(latest_rsi, 2),
            "Latest Close": round(latest_close, 2),
            "EPS": round(eps, 4) if not np.isnan(eps) else "N/A",
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

# ------------------------------
# Calculate Position Sizing
# ------------------------------
TOTAL_CAPITAL = 1000000
total_trades = results_df["Number of Trades"].sum()

if total_trades > 0:
    capital_per_trade = TOTAL_CAPITAL / total_trades
else:
    capital_per_trade = 0

if buy_signals:
    buy_df = pd.DataFrame(buy_signals, columns=["Ticker", "Close Price", "RSI", "EPS"])
    buy_df["Close Price"] = buy_df["Close Price"].round(2)
    buy_df["RSI"]         = buy_df["RSI"].round(2)
    buy_df["EPS"]         = buy_df["EPS"].round(4)
    buy_df["Order Size"]  = (capital_per_trade / buy_df["Close Price"]).apply(lambda x: int(round(x)))
else:
    buy_df = pd.DataFrame()

if sell_signals:
    sell_df = pd.DataFrame(sell_signals, columns=["Ticker", "Close Price", "RSI", "EPS"])
    sell_df["Close Price"] = sell_df["Close Price"].round(2)
    sell_df["RSI"]         = sell_df["RSI"].round(2)
    sell_df["EPS"]         = sell_df["EPS"].apply(lambda x: round(x, 4) if not np.isnan(x) else "N/A")
    sell_df["Order Size"]  = (capital_per_trade / sell_df["Close Price"]).apply(lambda x: int(round(x)))
else:
    sell_df = pd.DataFrame()

# ------------------------------
# Display Results
# ------------------------------
st.subheader("📈 RSI Strategy Results (BIST30)")
st.dataframe(results_df, use_container_width=True)

st.info(f"💰 **Capital Allocation:** Total Capital = ₺{TOTAL_CAPITAL:,.0f} | Total Trades = {total_trades} | Capital per Trade = ₺{capital_per_trade:,.2f}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🟢 Current BUY Signals (EPS > 0)")
    if not buy_df.empty:
        st.dataframe(buy_df, use_container_width=True)
    else:
        st.info("No buy signals with positive EPS found.")

with col2:
    st.subheader("🔴 Current SELL Signals")
    if not sell_df.empty:
        st.dataframe(sell_df, use_container_width=True)
    else:
        st.info("No sell signals found.")

# ================================================================
# RSI GRID — one small chart per stock, 3 per row
# ================================================================
st.subheader("📉 RSI Overview — All Stocks")
st.caption(
    "Green dashed line = buy threshold · Red dashed line = sell threshold · "
    "🟢 BUY  🔴 SELL  ⚪ HOLD"
)

SIGNAL_COLORS = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#94a3b8"}
SIGNAL_EMOJI  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}

COLS_PER_ROW = 3
ticker_list  = list(rsi_store.keys())
n_tickers    = len(ticker_list)
n_rows       = math.ceil(n_tickers / COLS_PER_ROW)

fig, axes = plt.subplots(
    n_rows, COLS_PER_ROW,
    figsize=(5 * COLS_PER_ROW, 3 * n_rows),
    facecolor="#0f172a"
)
fig.subplots_adjust(hspace=0.55, wspace=0.35)

axes_flat = axes.flatten() if n_tickers > 1 else [axes]

for i, ticker in enumerate(ticker_list):
    ax  = axes_flat[i]
    rsi = rsi_store[ticker]["rsi"]
    sig = rsi_store[ticker]["signal"]
    latest_rsi_val = float(rsi.iloc[-1])

    ax.set_facecolor("#1e293b")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=6)

    line_color = SIGNAL_COLORS[sig]
    ax.plot(rsi.values, color=line_color, lw=1.4)

    ax.axhline(35, color="#22c55e", lw=0.9, ls="--", alpha=0.8)
    ax.axhline(65, color="#ef4444", lw=0.9, ls="--", alpha=0.8)

    x_range = np.arange(len(rsi))
    ax.fill_between(x_range, rsi.values, 35,
                    where=(rsi.values < 35), color="#22c55e", alpha=0.15)
    ax.fill_between(x_range, rsi.values, 65,
                    where=(rsi.values > 65), color="#ef4444", alpha=0.15)

    ax.text(len(rsi) - 1, 35, " 35", color="#22c55e", fontsize=5.5, va="center", ha="left")
    ax.text(len(rsi) - 1, 65, " 65", color="#ef4444", fontsize=5.5, va="center", ha="left")

    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(rsi) - 1)
    ax.set_yticks([0, 35, 50, 65, 100])
    ax.set_yticklabels(["0", "35", "50", "65", "100"], fontsize=5.5, color="#94a3b8")
    ax.set_xticks([])
    ax.grid(color="#334155", alpha=0.4, lw=0.5)

    ax.set_title(
        f"{SIGNAL_EMOJI[sig]} {ticker}   RSI={latest_rsi_val:.1f}",
        color=SIGNAL_COLORS[sig], fontsize=7.5, fontweight="bold", pad=4
    )

for j in range(n_tickers, len(axes_flat)):
    axes_flat[j].set_visible(False)

st.pyplot(fig)
plt.close(fig)


# ================================================================
# PRICE & MOVING AVERAGES GRID — last 252 days, 3 per row
# ================================================================
st.subheader("📈 Close Price & Moving Averages — Last 252 Days")
st.caption(
    "⚪ Close Price  ·  🟡 MA50  ·  🔵 MA200  ·  Title colour = current signal"
)

MA50_COLOR  = "#facc15"   # vivid amber / gold
MA200_COLOR = "#22d3ee"   # electric cyan

price_ticker_list = list(price_store.keys())
n_price           = len(price_ticker_list)
n_price_rows      = math.ceil(n_price / COLS_PER_ROW)

fig2, axes2 = plt.subplots(
    n_price_rows, COLS_PER_ROW,
    figsize=(5 * COLS_PER_ROW, 3 * n_price_rows),
    facecolor="#0f172a"
)
fig2.subplots_adjust(hspace=0.55, wspace=0.35)

axes2_flat = axes2.flatten() if n_price > 1 else [axes2]

for i, ticker in enumerate(price_ticker_list):
    ax    = axes2_flat[i]
    close = price_store[ticker]["close"]
    ma50  = price_store[ticker]["ma50"]
    ma200 = price_store[ticker]["ma200"]
    sig   = price_store[ticker]["signal"]

    x          = np.arange(len(close))
    close_vals = close.values.astype(float)
    ma50_vals  = ma50.values.astype(float)
    ma200_vals = ma200.values.astype(float)
    valid50    = ~np.isnan(ma50_vals)
    valid200   = ~np.isnan(ma200_vals)

    # ---- axes styling ----
    ax.set_facecolor("#0d1b2a")
    for spine in ax.spines.values():
        spine.set_color("#1e3a5f")
    ax.tick_params(colors="#64748b", labelsize=5.5)
    ax.grid(color="#1e3a5f", alpha=0.45, lw=0.5)

    # ---- close price: white ----
    ax.plot(x, close_vals, color="#ffffff", lw=1.2, zorder=3, label="Close")
    ax.fill_between(x, close_vals, np.nanmin(close_vals),
                    color="#ffffff", alpha=0.04, zorder=2)

    # ---- MA50: gold ----
    if valid50.any():
        ax.plot(x[valid50], ma50_vals[valid50],
                color=MA50_COLOR, lw=1.4, zorder=4, label="MA50")

    # ---- MA200: cyan ----
    if valid200.any():
        ax.plot(x[valid200], ma200_vals[valid200],
                color=MA200_COLOR, lw=1.4, zorder=4, label="MA200")

    # ---- y-axis limits ----
    all_vals = np.concatenate([
        close_vals,
        ma50_vals[valid50]   if valid50.any()  else np.array([]),
        ma200_vals[valid200] if valid200.any() else np.array([])
    ])
    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
    pad = (ymax - ymin) * 0.08 if ymax != ymin else 1
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(0, len(close) - 1)
    ax.set_xticks([])

    # ---- latest price label ----
    latest_price = float(close_vals[-1])
    ax.text(len(close) - 1, latest_price,
            f" {latest_price:,.1f}",
            color="#ffffff", fontsize=5, va="center", ha="left", zorder=5)

    # ---- compact legend ----
    ax.legend(
        fontsize=5, loc="upper left",
        facecolor="#0d1b2a", edgecolor="#1e3a5f",
        labelcolor="#cbd5e1", framealpha=0.85,
        handlelength=1.4, handletextpad=0.4,
        borderpad=0.4, labelspacing=0.25
    )

    # ---- title ----
    ax.set_title(
        f"{SIGNAL_EMOJI[sig]} {ticker}   ₺{latest_price:,.1f}",
        color=SIGNAL_COLORS[sig], fontsize=7.5, fontweight="bold", pad=4
    )

for j in range(n_price, len(axes2_flat)):
    axes2_flat[j].set_visible(False)

st.pyplot(fig2)
plt.close(fig2)


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
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data["RSI"] = compute_rsi(data["Close"], rsi_period)
    data = data.dropna()

    forecast = lstm_forecast_rsi(data["RSI"], n_past=9, n_future=4)

    forecast_df = pd.DataFrame({
        "Ticker":    [selected_ticker],
        "Day+1 RSI": [round(forecast[0], 2)],
        "Day+2 RSI": [round(forecast[1], 2)],
        "Day+3 RSI": [round(forecast[2], 2)],
        "Day+4 RSI": [round(forecast[3], 2)],
    })
    st.dataframe(forecast_df, use_container_width=True)

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








# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="BIST30 Technical & Fundamental Strategy with Machine Forecast", layout="wide")

# # ------------------------------
# # Title
# # ------------------------------
# st.title("📊 BIST30 Technical Strategy — Backtest & LSTM Forecast")

# # ------------------------------
# # Sidebar Parameters
# # ------------------------------
# st.sidebar.header("🔧 Strategy Parameters")

# period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "3y"], index=1)
# rsi_period = st.sidebar.slider("RSI Period", 5, 30, 9)
# buy_threshold = st.sidebar.slider("Buy Threshold (RSI < x1)", 5, 45, 40)
# sell_threshold = st.sidebar.slider("Sell Threshold (RSI > x2)", 55, 95, 63)
# tcost = st.sidebar.number_input("Transaction Cost (e.g., 0.002 = 0.2%)", value=0.002, step=0.0005)

# # ------------------------------
# # Define BIST30 tickers
# # ------------------------------
# bist30_tickers = [
#     "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS", "EKGYO.IS", "ENJSA.IS", "EREGL.IS",
#     "FROTO.IS", "GARAN.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS", "KOZAL.IS",
#     "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
#     "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "TTRAK.IS", "HALKB.IS", "ALARK.IS"
# ]

# # ------------------------------
# # EPS Function
# # ------------------------------
# def get_eps(ticker):
#     """Fetch EPS (Earnings Per Share) for a given ticker"""
#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info
#         eps = info.get('trailingEps', None)
#         return eps if eps is not None else np.nan
#     except:
#         return np.nan

# # ------------------------------
# # RSI Function
# # ------------------------------
# def compute_rsi(series, period=14):
#     delta = series.diff()
#     gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
#     loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))

# # ------------------------------
# # Backtest Function
# # ------------------------------
# def backtest_strategy(df, x1, x2, tcost):
#     open_positions = []
#     closed_trades = []
#     for i in range(1, len(df)):
#         rsi = df["RSI"].iloc[i]
#         price = df["Close"].iloc[i]
#         date = df.index[i]

#         if rsi < x1:
#             open_positions.append({"entry_price": price, "entry_date": date})
#         elif rsi > x2 and open_positions:
#             entry = open_positions.pop(0)
#             closed_trades.append({
#                 "buy_date": entry["entry_date"],
#                 "buy_price": entry["entry_price"],
#                 "sell_date": date,
#                 "sell_price": price,
#                 "return": (price - entry["entry_price"]) / entry["entry_price"] - tcost
#             })

#     total_return = np.sum([t["return"] for t in closed_trades])
#     avg_return = np.mean([t["return"] for t in closed_trades]) if closed_trades else 0
#     return total_return, avg_return, closed_trades

# # ------------------------------
# # Analysis Loop
# # ------------------------------
# st.subheader("🔍 Scanning BIST30 Stocks...")

# results = []
# buy_signals = []
# sell_signals = []

# for ticker in bist30_tickers:
#     try:
#         data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
#         if data.empty:
#             continue
#         data["RSI"] = compute_rsi(data["Close"], rsi_period)
#         data = data.dropna()

#         total_return, avg_return, trades = backtest_strategy(data, buy_threshold, sell_threshold, tcost)
#         latest_rsi = float(data["RSI"].iloc[-1])
#         latest_close = float(data["Close"].iloc[-1])
        
#         # Fetch EPS
#         eps = get_eps(ticker)

#         if latest_rsi < buy_threshold:
#             # Only add to buy signals if EPS is positive
#             if not np.isnan(eps) and eps > 0:
#                 signal = "BUY"
#                 buy_signals.append((ticker, latest_close, latest_rsi, eps))
#             else:
#                 signal = "HOLD"
#         elif latest_rsi > sell_threshold:
#             signal = "SELL"
#             sell_signals.append((ticker, latest_close, latest_rsi, eps))
#         else:
#             signal = "HOLD"

#         results.append({
#             "Ticker": ticker,
#             "Signal": signal,
#             "Latest RSI": round(latest_rsi, 2),
#             "Latest Close": round(latest_close, 2),
#             "EPS": round(eps, 4) if not np.isnan(eps) else "N/A",
#             "Cumulative Return (%)": round(total_return * 100, 2),
#             "Return per Trade (%)": round(avg_return * 100, 2),
#             "Number of Trades": len(trades)
#         })

#     except Exception as e:
#         print(f"Error with {ticker}: {e}")

# # ------------------------------
# # Convert to DataFrame
# # ------------------------------
# results_df = pd.DataFrame(results).sort_values(by="Return per Trade (%)", ascending=False)

# # ------------------------------
# # Calculate Position Sizing
# # ------------------------------
# TOTAL_CAPITAL = 1000000  # Total capital in Liras
# total_trades = results_df["Number of Trades"].sum()

# if total_trades > 0:
#     capital_per_trade = TOTAL_CAPITAL / total_trades
# else:
#     capital_per_trade = 0

# # Format buy and sell DataFrames with proper rounding and order size
# if buy_signals:
#     buy_df = pd.DataFrame(buy_signals, columns=["Ticker", "Close Price", "RSI", "EPS"])
#     buy_df["Close Price"] = buy_df["Close Price"].round(2)
#     buy_df["RSI"] = buy_df["RSI"].round(2)
#     buy_df["EPS"] = buy_df["EPS"].round(4)
#     # Calculate order size (number of shares)
#     buy_df["Order Size"] = (capital_per_trade / buy_df["Close Price"]).apply(lambda x: int(round(x)))
# else:
#     buy_df = pd.DataFrame()

# if sell_signals:
#     sell_df = pd.DataFrame(sell_signals, columns=["Ticker", "Close Price", "RSI", "EPS"])
#     sell_df["Close Price"] = sell_df["Close Price"].round(2)
#     sell_df["RSI"] = sell_df["RSI"].round(2)
#     sell_df["EPS"] = sell_df["EPS"].apply(lambda x: round(x, 4) if not np.isnan(x) else "N/A")
#     # Calculate order size (number of shares)
#     sell_df["Order Size"] = (capital_per_trade / sell_df["Close Price"]).apply(lambda x: int(round(x)))
# else:
#     sell_df = pd.DataFrame()

# # ------------------------------
# # Display Results
# # ------------------------------
# st.subheader("📈 RSI Strategy Results (BIST30)")
# st.dataframe(results_df, use_container_width=True)

# # Display capital allocation info
# st.info(f"💰 **Capital Allocation:** Total Capital = ₺{TOTAL_CAPITAL:,.0f} | Total Trades = {total_trades} | Capital per Trade = ₺{capital_per_trade:,.2f}")

# col1, col2 = st.columns(2)
# with col1:
#     st.subheader("🟢 Current BUY Signals (EPS > 0)")
#     if not buy_df.empty:
#         st.dataframe(buy_df, use_container_width=True)
#     else:
#         st.info("No buy signals with positive EPS found.")

# with col2:
#     st.subheader("🔴 Current SELL Signals")
#     if not sell_df.empty:
#         st.dataframe(sell_df, use_container_width=True)
#     else:
#         st.info("No sell signals found.")

# # ================================================================
# # PART 2: Select Stock for LSTM Forecast
# # ================================================================
# st.subheader("🤖 LSTM RSI Forecast (User-Selected Stock)")

# selected_ticker = st.selectbox("Select a stock for RSI forecast:", ["None"] + bist30_tickers)

# # ------------------------------
# # LSTM Function
# # ------------------------------
# def lstm_forecast_rsi(rsi_series, n_past=9, n_future=4):
#     if len(rsi_series) < n_past + 5:
#         return [np.nan] * n_future

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     rsi_scaled = scaler.fit_transform(rsi_series.values.reshape(-1, 1))

#     X, y = [], []
#     for i in range(n_past, len(rsi_scaled) - n_future):
#         X.append(rsi_scaled[i - n_past:i, 0])
#         y.append(rsi_scaled[i:i + n_future, 0])
#     X, y = np.array(X), np.array(y)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     model = Sequential([
#         LSTM(50, activation='relu', input_shape=(n_past, 1)),
#         Dense(25, activation='relu'),
#         Dense(n_future)
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, y, epochs=30, batch_size=8, verbose=0)

#     last_window = rsi_scaled[-n_past:].reshape((1, n_past, 1))
#     forecast_scaled = model.predict(last_window)
#     forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
#     return forecast

# # ------------------------------
# # Run forecast only if user selected a stock
# # ------------------------------
# if selected_ticker != "None":
#     st.write(f"### 🔮 Forecasting RSI for: **{selected_ticker}**")
#     data = yf.download(selected_ticker, period=period, auto_adjust=True, progress=False)
#     data["RSI"] = compute_rsi(data["Close"], rsi_period)
#     data = data.dropna()

#     forecast = lstm_forecast_rsi(data["RSI"], n_past=9, n_future=4)

#     # Show forecast table
#     forecast_df = pd.DataFrame({
#         "Ticker": [selected_ticker],
#         "Day+1 RSI": [round(forecast[0], 2)],
#         "Day+2 RSI": [round(forecast[1], 2)],
#         "Day+3 RSI": [round(forecast[2], 2)],
#         "Day+4 RSI": [round(forecast[3], 2)],
#     })
#     st.dataframe(forecast_df, use_container_width=True)

#     # Mini plot for Close price
#     st.write("📉 Recent Close Price Trend")
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.plot(data.index[-100:], data["Close"].iloc[-100:], label="Close", color="steelblue")
#     ax.set_title(f"{selected_ticker} — Recent Close Prices")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Close Price")
#     ax.legend()
#     st.pyplot(fig)

# else:
#     st.info("Select a stock above to generate RSI LSTM forecast.")

# st.caption("Developed for educational and research purposes — RSI Strategy + LSTM Forecast on BIST30.")










# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="BIST30 Technical & Fundamental Strategy with Machine Forecast", layout="wide")

# # ------------------------------
# # Title
# # ------------------------------
# st.title("📊 BIST30 Technical Strategy — Backtest & LSTM Forecast")

# # ------------------------------
# # Sidebar Parameters
# # ------------------------------
# st.sidebar.header("🔧 Strategy Parameters")

# period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "3y"], index=1)
# rsi_period = st.sidebar.slider("RSI Period", 5, 30, 9)
# buy_threshold = st.sidebar.slider("Buy Threshold (RSI < x1)", 5, 45, 40)
# sell_threshold = st.sidebar.slider("Sell Threshold (RSI > x2)", 55, 95, 63)
# tcost = st.sidebar.number_input("Transaction Cost (e.g., 0.002 = 0.2%)", value=0.002, step=0.0005)

# # ------------------------------
# # Define BIST30 tickers
# # ------------------------------
# bist30_tickers = [
#     "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS", "EKGYO.IS", "ENJSA.IS", "EREGL.IS",
#     "FROTO.IS", "GARAN.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS", "KOZAL.IS",
#     "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
#     "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "TTRAK.IS", "HALKB.IS", "ALARK.IS"
# ]

# # ------------------------------
# # EPS Function
# # ------------------------------
# def get_eps(ticker):
#     """Fetch EPS (Earnings Per Share) for a given ticker"""
#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info
#         eps = info.get('trailingEps', None)
#         return eps if eps is not None else np.nan
#     except:
#         return np.nan

# # ------------------------------
# # RSI Function
# # ------------------------------
# def compute_rsi(series, period=14):
#     delta = series.diff()
#     gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
#     loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))

# # ------------------------------
# # Backtest Function
# # ------------------------------
# def backtest_strategy(df, x1, x2, tcost):
#     open_positions = []
#     closed_trades = []
#     for i in range(1, len(df)):
#         rsi = df["RSI"].iloc[i]
#         price = df["Close"].iloc[i]
#         date = df.index[i]

#         if rsi < x1:
#             open_positions.append({"entry_price": price, "entry_date": date})
#         elif rsi > x2 and open_positions:
#             entry = open_positions.pop(0)
#             closed_trades.append({
#                 "buy_date": entry["entry_date"],
#                 "buy_price": entry["entry_price"],
#                 "sell_date": date,
#                 "sell_price": price,
#                 "return": (price - entry["entry_price"]) / entry["entry_price"] - tcost
#             })

#     total_return = np.sum([t["return"] for t in closed_trades])
#     avg_return = np.mean([t["return"] for t in closed_trades]) if closed_trades else 0
#     return total_return, avg_return, closed_trades

# # ------------------------------
# # Analysis Loop
# # ------------------------------
# st.subheader("🔍 Scanning BIST30 Stocks...")

# results = []
# buy_signals = []
# sell_signals = []

# for ticker in bist30_tickers:
#     try:
#         data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
#         if data.empty:
#             continue
#         data["RSI"] = compute_rsi(data["Close"], rsi_period)
#         data = data.dropna()

#         total_return, avg_return, trades = backtest_strategy(data, buy_threshold, sell_threshold, tcost)
#         latest_rsi = float(data["RSI"].iloc[-1])
#         latest_close = float(data["Close"].iloc[-1])
        
#         # Fetch EPS
#         eps = get_eps(ticker)

#         if latest_rsi < buy_threshold:
#             # Only add to buy signals if EPS is positive
#             if not np.isnan(eps) and eps > 0:
#                 signal = "BUY"
#                 buy_signals.append((ticker, latest_close, latest_rsi, eps))
#             else:
#                 signal = "HOLD"
#         elif latest_rsi > sell_threshold:
#             signal = "SELL"
#             sell_signals.append((ticker, latest_close, latest_rsi, eps))
#         else:
#             signal = "HOLD"

#         results.append({
#             "Ticker": ticker,
#             "Signal": signal,
#             "Latest RSI": round(latest_rsi, 2),
#             "Latest Close": round(latest_close, 2),
#             "EPS": round(eps, 4) if not np.isnan(eps) else "N/A",
#             "Cumulative Return (%)": round(total_return * 100, 2),
#             "Return per Trade (%)": round(avg_return * 100, 2),
#             "Number of Trades": len(trades)
#         })

#     except Exception as e:
#         print(f"Error with {ticker}: {e}")

# # ------------------------------
# # Convert to DataFrame
# # ------------------------------
# results_df = pd.DataFrame(results).sort_values(by="Return per Trade (%)", ascending=False)
# buy_df = pd.DataFrame(buy_signals, columns=["Ticker", "Close Price", "RSI", "EPS"])
# sell_df = pd.DataFrame(sell_signals, columns=["Ticker", "Close Price", "RSI", "EPS"])

# # ------------------------------
# # Display Results
# # ------------------------------
# st.subheader("📈 RSI Strategy Results (BIST30)")
# st.dataframe(results_df, use_container_width=True)

# col1, col2 = st.columns(2)
# with col1:
#     st.subheader("🟢 Current BUY Signals (EPS > 0)")
#     if not buy_df.empty:
#         buy_df["EPS"] = buy_df["EPS"].round(4)
#         st.dataframe(buy_df, use_container_width=True)
#     else:
#         st.dataframe(pd.DataFrame([["-", "-", "-", "-"]], columns=["Ticker","Close Price","RSI", "EPS"]))

# with col2:
#     st.subheader("🔴 Current SELL Signals")
#     if not sell_df.empty:
#         sell_df["EPS"] = sell_df["EPS"].apply(lambda x: round(x, 4) if not np.isnan(x) else "N/A")
#         st.dataframe(sell_df, use_container_width=True)
#     else:
#         st.dataframe(pd.DataFrame([["-", "-", "-", "-"]], columns=["Ticker","Close Price","RSI", "EPS"]))

# # ================================================================
# # PART 2: Select Stock for LSTM Forecast
# # ================================================================
# st.subheader("🤖 LSTM RSI Forecast (User-Selected Stock)")

# selected_ticker = st.selectbox("Select a stock for RSI forecast:", ["None"] + bist30_tickers)

# # ------------------------------
# # LSTM Function
# # ------------------------------
# def lstm_forecast_rsi(rsi_series, n_past=9, n_future=4):
#     if len(rsi_series) < n_past + 5:
#         return [np.nan] * n_future

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     rsi_scaled = scaler.fit_transform(rsi_series.values.reshape(-1, 1))

#     X, y = [], []
#     for i in range(n_past, len(rsi_scaled) - n_future):
#         X.append(rsi_scaled[i - n_past:i, 0])
#         y.append(rsi_scaled[i:i + n_future, 0])
#     X, y = np.array(X), np.array(y)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     model = Sequential([
#         LSTM(50, activation='relu', input_shape=(n_past, 1)),
#         Dense(25, activation='relu'),
#         Dense(n_future)
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, y, epochs=30, batch_size=8, verbose=0)

#     last_window = rsi_scaled[-n_past:].reshape((1, n_past, 1))
#     forecast_scaled = model.predict(last_window)
#     forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
#     return forecast

# # ------------------------------
# # Run forecast only if user selected a stock
# # ------------------------------
# if selected_ticker != "None":
#     st.write(f"### 🔮 Forecasting RSI for: **{selected_ticker}**")
#     data = yf.download(selected_ticker, period=period, auto_adjust=True, progress=False)
#     data["RSI"] = compute_rsi(data["Close"], rsi_period)
#     data = data.dropna()

#     forecast = lstm_forecast_rsi(data["RSI"], n_past=9, n_future=4)

#     # Show forecast table
#     forecast_df = pd.DataFrame({
#         "Ticker": [selected_ticker],
#         "Day+1 RSI": [round(forecast[0], 2)],
#         "Day+2 RSI": [round(forecast[1], 2)],
#         "Day+3 RSI": [round(forecast[2], 2)],
#         "Day+4 RSI": [round(forecast[3], 2)],
#     })
#     st.dataframe(forecast_df, use_container_width=True)

#     # Mini plot for Close price
#     st.write("📉 Recent Close Price Trend")
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.plot(data.index[-100:], data["Close"].iloc[-100:], label="Close", color="steelblue")
#     ax.set_title(f"{selected_ticker} — Recent Close Prices")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Close Price")
#     ax.legend()
#     st.pyplot(fig)

# else:
#     st.info("Select a stock above to generate RSI LSTM forecast.")

# st.caption("Developed for educational and research purposes — RSI Strategy + LSTM Forecast on BIST30.")





# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="BIST30 Technical & Fundamental Strategy with Machine Forecast", layout="wide")

# # ------------------------------
# # Title
# # ------------------------------
# st.title("📊 BIST30 Technical Strategy — Backtest & LSTM Forecast")

# # ------------------------------
# # Sidebar Parameters
# # ------------------------------
# st.sidebar.header("🔧 Strategy Parameters")

# period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "3y"], index=1)
# rsi_period = st.sidebar.slider("RSI Period", 5, 30, 9)
# buy_threshold = st.sidebar.slider("Buy Threshold (RSI < x1)", 5, 45, 40)
# sell_threshold = st.sidebar.slider("Sell Threshold (RSI > x2)", 55, 95, 63)
# tcost = st.sidebar.number_input("Transaction Cost (e.g., 0.002 = 0.2%)", value=0.002, step=0.0005)

# # ------------------------------
# # Define BIST30 tickers
# # ------------------------------
# bist30_tickers = [
#     "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS", "EKGYO.IS", "ENJSA.IS", "EREGL.IS",
#     "FROTO.IS", "GARAN.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS", "KOZAL.IS",
#     "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
#     "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "TTRAK.IS", "HALKB.IS", "ALARK.IS"
# ]

# # ------------------------------
# # RSI Function
# # ------------------------------
# def compute_rsi(series, period=14):
#     delta = series.diff()
#     gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
#     loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))

# # ------------------------------
# # Backtest Function
# # ------------------------------
# def backtest_strategy(df, x1, x2, tcost):
#     open_positions = []
#     closed_trades = []
#     for i in range(1, len(df)):
#         rsi = df["RSI"].iloc[i]
#         price = df["Close"].iloc[i]
#         date = df.index[i]

#         if rsi < x1:
#             open_positions.append({"entry_price": price, "entry_date": date})
#         elif rsi > x2 and open_positions:
#             entry = open_positions.pop(0)
#             closed_trades.append({
#                 "buy_date": entry["entry_date"],
#                 "buy_price": entry["entry_price"],
#                 "sell_date": date,
#                 "sell_price": price,
#                 "return": (price - entry["entry_price"]) / entry["entry_price"] - tcost
#             })

#     total_return = np.sum([t["return"] for t in closed_trades])
#     avg_return = np.mean([t["return"] for t in closed_trades]) if closed_trades else 0
#     return total_return, avg_return, closed_trades

# # ------------------------------
# # Analysis Loop
# # ------------------------------
# st.subheader("🔍 Scanning BIST30 Stocks...")

# results = []
# buy_signals = []
# sell_signals = []

# for ticker in bist30_tickers:
#     try:
#         data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
#         if data.empty:
#             continue
#         data["RSI"] = compute_rsi(data["Close"], rsi_period)
#         data = data.dropna()

#         total_return, avg_return, trades = backtest_strategy(data, buy_threshold, sell_threshold, tcost)
#         latest_rsi = float(data["RSI"].iloc[-1])
#         latest_close = float(data["Close"].iloc[-1])

#         if latest_rsi < buy_threshold:
#             signal = "BUY"
#             buy_signals.append((ticker, latest_close, latest_rsi))
#         elif latest_rsi > sell_threshold:
#             signal = "SELL"
#             sell_signals.append((ticker, latest_close, latest_rsi))
#         else:
#             signal = "HOLD"

#         results.append({
#             "Ticker": ticker,
#             "Signal": signal,
#             "Latest RSI": round(latest_rsi, 2),
#             "Latest Close": round(latest_close, 2),
#             "Cumulative Return (%)": round(total_return * 100, 2),
#             "Return per Trade (%)": round(avg_return * 100, 2),
#             "Number of Trades": len(trades)
#         })

#     except Exception as e:
#         print(f"Error with {ticker}: {e}")

# # ------------------------------
# # Convert to DataFrame
# # ------------------------------
# results_df = pd.DataFrame(results).sort_values(by="Return per Trade (%)", ascending=False)
# buy_df = pd.DataFrame(buy_signals, columns=["Ticker", "Close Price", "RSI"])
# sell_df = pd.DataFrame(sell_signals, columns=["Ticker", "Close Price", "RSI"])

# # ------------------------------
# # Display Results
# # ------------------------------
# st.subheader("📈 RSI Strategy Results (BIST30)")
# st.dataframe(results_df, use_container_width=True)

# col1, col2 = st.columns(2)
# with col1:
#     st.subheader("🟢 Current BUY Signals")
#     st.dataframe(buy_df if not buy_df.empty else pd.DataFrame([["-", "-", "-"]], columns=["Ticker","Close Price","RSI"]))

# with col2:
#     st.subheader("🔴 Current SELL Signals")
#     st.dataframe(sell_df if not sell_df.empty else pd.DataFrame([["-", "-", "-"]], columns=["Ticker","Close Price","RSI"]))

# # ================================================================
# # PART 2: Select Stock for LSTM Forecast
# # ================================================================
# st.subheader("🤖 LSTM RSI Forecast (User-Selected Stock)")

# selected_ticker = st.selectbox("Select a stock for RSI forecast:", ["None"] + bist30_tickers)

# # ------------------------------
# # LSTM Function
# # ------------------------------
# def lstm_forecast_rsi(rsi_series, n_past=9, n_future=4):
#     if len(rsi_series) < n_past + 5:
#         return [np.nan] * n_future

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     rsi_scaled = scaler.fit_transform(rsi_series.values.reshape(-1, 1))

#     X, y = [], []
#     for i in range(n_past, len(rsi_scaled) - n_future):
#         X.append(rsi_scaled[i - n_past:i, 0])
#         y.append(rsi_scaled[i:i + n_future, 0])
#     X, y = np.array(X), np.array(y)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     model = Sequential([
#         LSTM(50, activation='relu', input_shape=(n_past, 1)),
#         Dense(25, activation='relu'),
#         Dense(n_future)
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, y, epochs=30, batch_size=8, verbose=0)

#     last_window = rsi_scaled[-n_past:].reshape((1, n_past, 1))
#     forecast_scaled = model.predict(last_window)
#     forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
#     return forecast

# # ------------------------------
# # Run forecast only if user selected a stock
# # ------------------------------
# if selected_ticker != "None":
#     st.write(f"### 🔮 Forecasting RSI for: **{selected_ticker}**")
#     data = yf.download(selected_ticker, period=period, auto_adjust=True, progress=False)
#     data["RSI"] = compute_rsi(data["Close"], rsi_period)
#     data = data.dropna()

#     forecast = lstm_forecast_rsi(data["RSI"], n_past=9, n_future=4)

#     # Show forecast table
#     forecast_df = pd.DataFrame({
#         "Ticker": [selected_ticker],
#         "Day+1 RSI": [round(forecast[0], 2)],
#         "Day+2 RSI": [round(forecast[1], 2)],
#         "Day+3 RSI": [round(forecast[2], 2)],
#         "Day+4 RSI": [round(forecast[3], 2)],
#     })
#     st.dataframe(forecast_df, use_container_width=True)

#     # Mini plot for Close price
#     st.write("📉 Recent Close Price Trend")
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.plot(data.index[-100:], data["Close"].iloc[-100:], label="Close", color="steelblue")
#     ax.set_title(f"{selected_ticker} — Recent Close Prices")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Close Price")
#     ax.legend()
#     st.pyplot(fig)

# else:
#     st.info("Select a stock above to generate RSI LSTM forecast.")

# st.caption("Developed for educational and research purposes — RSI Strategy + LSTM Forecast on BIST30.")
