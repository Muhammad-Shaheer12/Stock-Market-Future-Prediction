from __future__ import annotations
import numpy as np
import pandas as pd


def sharpe_ratio(returns, risk_free=0.0):
    r = np.array(returns) - risk_free
    if r.std() == 0:
        return 0.0
    return float(np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252))


def max_drawdown(cum_returns):
    arr = np.array(cum_returns)
    peaks = np.maximum.accumulate(arr)
    dd = (arr - peaks) / (peaks + 1e-9)
    return float(dd.min())


def backtest_prices(df: pd.DataFrame, pred_returns: pd.Series):
    # pred_returns are log-returns for next step
    # Shift prediction to align with next period
    strat = pred_returns.shift(1).fillna(0.0)
    realized = df["log_ret"].fillna(0.0)
    pnl = strat * realized
    cum = pnl.cumsum()
    return {
        "cum_return": float(cum.iloc[-1]),
        "sharpe": sharpe_ratio(pnl),
        "max_drawdown": max_drawdown(cum),
    }
