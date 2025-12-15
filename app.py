from __future__ import annotations

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import fetch_adj_close
from src.portfolio import normalize_weights, build_equity_curve
from src.metrics import cagr, annualized_vol, sharpe_ratio, max_drawdown, drawdown

st.set_page_config(page_title="Portfolio Analytics Dashboard", layout="wide")

st.title("📈 Portfolio Analytics Dashboard")
st.caption("CAGR • Volatility • Sharpe • Max Drawdown • Portfolio vs Benchmark (yfinance + Python)")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload portfolio CSV (ticker, weight)", type=["csv"])
    st.write("Or use the sample file format: `ticker,weight`")

    benchmark = st.text_input("Benchmark ticker (e.g., SPY, VOO, IWDA.L)", value="SPY").strip().upper()
    start = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    end = st.date_input("End date", value=pd.to_datetime("today"))
    risk_free = st.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.20, value=0.02, step=0.005)

    initial_value = st.number_input("Initial portfolio value", min_value=1.0, value=100.0, step=10.0)

if not uploaded:
    st.info("Upload a portfolio CSV to begin (columns: ticker, weight).")
    st.stop()

pf_df = pd.read_csv(uploaded)
pf_df = normalize_weights(pf_df)

tickers = pf_df["ticker"].tolist()
weights = dict(zip(pf_df["ticker"], pf_df["weight"]))
all_tickers = sorted(set(tickers + ([benchmark] if benchmark else [])))

prices = fetch_adj_close(all_tickers, start=str(start), end=str(end))

# Split prices
pf_prices = prices[[t for t in tickers if t in prices.columns]].dropna(how="all")
bm_prices = prices[[benchmark]].dropna(how="all") if benchmark in prices.columns else None

port_daily, port_equity = build_equity_curve(pf_prices, weights, initial_value=initial_value)

# Benchmark equity (normalized)
bm_daily = None
bm_equity = None
if bm_prices is not None and not bm_prices.empty:
    bm_daily = bm_prices[benchmark].pct_change().dropna()
    bm_equity = (1 + bm_daily).cumprod() * initial_value

# Align series for comparisons/plots
if bm_equity is not None:
    aligned = pd.concat([port_equity.rename("Portfolio"), bm_equity.rename("Benchmark")], axis=1).dropna()
else:
    aligned = pd.DataFrame({"Portfolio": port_equity}).dropna()

# Metrics
port_cagr = cagr(aligned["Portfolio"])
port_vol = annualized_vol(port_daily.loc[aligned.index])
port_sharpe = sharpe_ratio(port_daily.loc[aligned.index], risk_free_rate=risk_free)
port_mdd = max_drawdown(aligned["Portfolio"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("CAGR", f"{port_cagr*100:.2f}%")
col2.metric("Volatility (ann.)", f"{port_vol*100:.2f}%")
col3.metric("Sharpe", f"{port_sharpe:.2f}")
col4.metric("Max Drawdown", f"{port_mdd*100:.2f}%")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Equity Curve")
    fig = plt.figure()
    plt.plot(aligned.index, aligned["Portfolio"], label="Portfolio")
    if "Benchmark" in aligned.columns:
        plt.plot(aligned.index, aligned["Benchmark"], label="Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    st.pyplot(fig)

with right:
    st.subheader("Drawdown")
    dd = drawdown(aligned["Portfolio"])
    fig2 = plt.figure()
    plt.plot(dd.index, dd.values)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    st.pyplot(fig2)

st.subheader("Holdings")
st.dataframe(pf_df, use_container_width=True)

st.subheader("Price Data (head)")
st.dataframe(prices.head(), use_container_width=True)
