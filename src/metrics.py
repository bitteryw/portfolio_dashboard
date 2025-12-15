from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def cagr(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return np.nan
    start_val = float(equity_curve.iloc[0])
    end_val = float(equity_curve.iloc[-1])
    if start_val <= 0:
        return np.nan

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    if years <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1


def annualized_vol(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return np.nan
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    risk_free_rate is annual (e.g., 0.02 = 2%).
    Uses simple conversion to daily rf.
    """
    if daily_returns.empty:
        return np.nan
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = daily_returns - rf_daily
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float((excess.mean() / vol) * np.sqrt(TRADING_DAYS))


def drawdown(equity_curve: pd.Series) -> pd.Series:
    if equity_curve.empty:
        return pd.Series(dtype=float)
    peak = equity_curve.cummax()
    dd = (equity_curve / peak) - 1.0
    return dd


def max_drawdown(equity_curve: pd.Series) -> float:
    dd = drawdown(equity_curve)
    if dd.empty:
        return np.nan
    return float(dd.min())
