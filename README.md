# Portfolio Analytics Dashboard (Python + yfinance + Streamlit)

A simple portfolio analytics dashboard that computes:
- CAGR
- Annualized Volatility
- Sharpe Ratio (with configurable risk-free rate)
- Max Drawdown + drawdown series
- Portfolio vs Benchmark equity curve

## Demo
Run locally via Streamlit.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate   # windows

pip install -r requirements.txt
