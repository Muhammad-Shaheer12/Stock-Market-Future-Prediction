from __future__ import annotations
from typing import List, Dict
import pandas as pd
from src.data.alpha import AlphaVantageClient
from src.features.engineering import json_to_ohlcv, compute_features

DEFAULT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "SPY"]


def fetch_universe(symbols: List[str] | None = None) -> Dict[str, pd.DataFrame]:
    symbols = symbols or DEFAULT_UNIVERSE
    client = AlphaVantageClient()
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        js = client.daily_adjusted(sym, outputsize="compact")
        df = json_to_ohlcv(js)
        df = compute_features(df)
        out[sym] = df
    return out
