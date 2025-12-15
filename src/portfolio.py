from __future__ import annotations

import numpy as np
import pandas as pd

from .data import to_daily_returns


def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns: ticker, weight
    """
    if not {"ticker", "weight"}.issubset(df.columns):
        raise ValueError("Portfolio file must have columns: ticker, weight")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["ticker", "weight"])
    s = df["weight"].sum()
    if s <= 0:
        raise ValueError("Sum of weights must be > 0")
    df["weight"] = df["weight"] / s
    return df


def build_equity_curve(prices: pd.DataFrame, weights: dict[str, float], initial_value: float = 100.0):
    """
    Builds portfolio equity curve using daily rebalanced weights (approx).
    """
    rets = to_daily_returns(prices)
    common = [t for t in rets.columns if t in weights]
    if not common:
        raise ValueError("No overlap between price tickers and weights")

    w = np.array([weights[t] for t in common], dtype=float)
    w = w / w.sum()

    port_daily = (rets[common] * w).sum(axis=1)
    equity = (1 + port_daily).cumprod() * initial_value
    return port_daily, equity
