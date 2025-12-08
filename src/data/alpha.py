from __future__ import annotations
import time
import json
from typing import Dict, Any
import requests
from .cache import cache_path
from src.utils.config import (
    ALPHAVANTAGE_API_KEY,
    ALPHAVANTAGE_RATE_LIMIT_S,
    REQUEST_TIMEOUT_S,
)

BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageClient:
    def __init__(self, api_key: str | None = None, rate_limit_s: float | None = None):
        self.api_key = api_key or ALPHAVANTAGE_API_KEY
        # Allow tests to run without a real key
        if not self.api_key and (
            __import__("os").environ.get("PYTEST_CURRENT_TEST")
            or __import__("os").environ.get("ALPHAVANTAGE_API_KEY")
        ):
            self.api_key = __import__("os").environ.get("ALPHAVANTAGE_API_KEY", "dummy")
        if not self.api_key:
            raise RuntimeError("Missing ALPHAVANTAGE_API_KEY")
        self.rate_limit_s = rate_limit_s or ALPHAVANTAGE_RATE_LIMIT_S
        self._last_call = 0.0

    def _throttle(self):
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self._last_call = time.time()

    def _valid_daily(self, js: Dict[str, Any]) -> bool:
        return (
            isinstance(js, dict)
            and "Time Series (Daily)" in js
            and isinstance(js["Time Series (Daily)"], dict)
            and len(js["Time Series (Daily)"]) > 0
        )

    def daily_adjusted(
        self, symbol: str, outputsize: str = "compact"
    ) -> Dict[str, Any]:
        key = f"daily_adjusted_{symbol}_{outputsize}.json"
        cp = cache_path(key)
        if cp.exists():
            with open(cp) as f:
                cached = json.load(f)
            if self._valid_daily(cached):
                return cached
            else:
                # invalidate bad cache and refetch
                try:
                    cp.unlink()
                except Exception:
                    pass
        self._throttle()
        params = {
            # Use non-premium daily data to avoid premium restrictions
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        # simple retry/backoff on transient failures
        last_exc = None
        for attempt in range(3):
            try:
                r = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT_S)
                r.raise_for_status()
                js = r.json()
                break
            except Exception as e:
                last_exc = e
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"AlphaVantage request failed: {last_exc}")
        # Only cache valid payloads; otherwise raise a helpful error
        if self._valid_daily(js):
            with open(cp, "w") as f:
                json.dump(js, f)
            return js
        # Prefer detailed Alpha Vantage message if present
        msg = (
            js.get("Note")
            or js.get("Information")
            or js.get("Error Message")
            or "Unexpected Alpha Vantage response"
        )
        raise RuntimeError(f"AlphaVantage error: {msg}")
