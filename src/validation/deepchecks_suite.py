from __future__ import annotations
import pandas as pd
from deepchecks.tabular.suites import data_integrity, model_evaluation
from deepchecks.tabular import Dataset
from typing import Any, Dict


def build_dataset(df: pd.DataFrame, label_col: str | None = None) -> Dataset:
    return Dataset(df, label=label_col, datetime_name="date")


def run_data_checks(df: pd.DataFrame) -> Dict[str, Any]:
    ds = build_dataset(df)
    suite = data_integrity()
    result = suite.run(ds)
    return {"passed": result.passed(), "summary": result.to_json()}


def run_model_checks(
    df: pd.DataFrame, model, features: list[str], label_col: str
) -> Dict[str, Any]:
    ds = build_dataset(df[features + [label_col]], label_col)
    suite = model_evaluation()
    result = suite.run(ds, model)
    return {"passed": result.passed(), "summary": result.to_json()}
