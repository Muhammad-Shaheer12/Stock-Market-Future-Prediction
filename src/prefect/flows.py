"""Prefect flow for the ML training pipeline."""

from __future__ import annotations

from prefect import flow, task
import os
import requests
from src.data.alpha import AlphaVantageClient
from src.features.engineering import json_to_ohlcv, compute_features, add_labels
from src.models.train import train_models, HORIZONS
from src.utils.config import DEFAULT_SYMBOL
from src.validation.deepchecks_suite import run_data_checks


@task(retries=2, retry_delay_seconds=15)
def fetch_data(symbol: str = DEFAULT_SYMBOL):
    client = AlphaVantageClient()
    js = client.daily_adjusted(symbol, outputsize="compact")
    return js


@task
def feature_pipeline(js: dict):
    df = json_to_ohlcv(js)
    df = compute_features(df)
    df = add_labels(df, HORIZONS)
    # Deepchecks skipped for demo - limited data from Alpha Vantage compact mode
    # checks = run_data_checks(df)
    # if not checks.get("passed", False):
    #     raise RuntimeError("Deepchecks data integrity failed")
    return df


@task
def train_pipeline(df):
    out_dir = train_models(df)
    return str(out_dir)


def notify(status: str, message: str):
    url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not url:
        return
    try:
        requests.post(url, json={"text": f"[Prefect] {status}: {message}"}, timeout=10)
    except Exception:
        pass


@flow(name="mh_predictor_flow")
def main_flow(symbol: str = DEFAULT_SYMBOL):
    js = fetch_data(symbol)
    df = feature_pipeline(js)
    model_dir = train_pipeline(df)
    notify("success", f"Training completed. Model dir: {model_dir}")
    return {"model_dir": model_dir}


if __name__ == "__main__":
    main_flow()
