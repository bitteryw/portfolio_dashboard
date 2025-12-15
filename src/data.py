from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch Adjusted Close prices for tickers between start and end (YYYY-MM-DD).
    Returns a DataFrame indexed by date, columns = tickers.
    """
    if not tickers:
        raise ValueError("tickers list is empty")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yf output shape differs for 1 ticker vs many
    if isinstance(data.columns, pd.MultiIndex):
        adj = pd.DataFrame({t: data[t]["Adj Close"] for t in tickers})
    else:
        # single ticker
        adj = pd.DataFrame({tickers[0]: data["Adj Close"]})

    adj = adj.dropna(how="all")
    return adj


def to_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from price series."""
    return prices.pct_change().dropna(how="all")
