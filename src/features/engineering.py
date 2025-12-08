from __future__ import annotations
import pandas as pd
import numpy as np

# Basic indicators without external libs


def json_to_ohlcv(js: dict) -> pd.DataFrame:
    ts = js.get("Time Series (Daily)", {})
    if not isinstance(ts, dict) or len(ts) == 0:
        raise ValueError("No Time Series (Daily) data in response")
    rows = []
    for ds, vals in ts.items():
        volume_val = vals.get("6. volume", vals.get("5. volume", "0"))
        adj_val = vals.get("5. adjusted close", vals.get("4. close"))
        rows.append(
            {
                "date": pd.to_datetime(ds),
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low": float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "adj_close": float(adj_val),
                "volume": float(volume_val),
            }
        )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = np.log(df["adj_close"]).diff()
    df["ret"] = df["adj_close"].pct_change()
    df["sma_5"] = df["adj_close"].rolling(5).mean()
    df["sma_10"] = df["adj_close"].rolling(10).mean()
    df["ema_10"] = df["adj_close"].ewm(span=10, adjust=False).mean()
    delta = df["adj_close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(14).mean()
    roll_down = pd.Series(down).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    ema12 = df["adj_close"].ewm(span=12, adjust=False).mean()
    ema26 = df["adj_close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd"] = macd
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    df["volatility_10"] = df["ret"].rolling(10).std()
    # Calendar features
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    return df


def add_labels(df: pd.DataFrame, horizons=(1, 3, 7, 30)) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        df[f"label_{h}"] = df["log_ret"].shift(-h).rolling(h).sum()
    return df


FEATURE_COLUMNS = [
    "ret",
    "log_ret",
    "sma_5",
    "sma_10",
    "ema_10",
    "rsi_14",
    "macd",
    "macd_signal",
    "volatility_10",
    "dow",
    "dom",
    "month",
]
