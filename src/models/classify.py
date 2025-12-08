from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from src.features.engineering import FEATURE_COLUMNS


@dataclass
class ClassificationResult:
    metrics: dict
    probs: float
    label: int


def build_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    future = df["adj_close"].shift(-horizon)
    return (future > df["adj_close"]).astype(int)


def train_and_predict(df: pd.DataFrame, horizon: int = 1) -> ClassificationResult:
    df = df.copy()
    df["label_cls"] = build_labels(df, horizon)
    d = df[FEATURE_COLUMNS + ["label_cls"]].dropna()
    if len(d) < 50:
        raise ValueError("Insufficient data for classification")
    split = int(len(d) * 0.8)
    train, test = d.iloc[:split], d.iloc[split:]
    Xtr, ytr = train[FEATURE_COLUMNS], train["label_cls"]
    Xte, yte = test[FEATURE_COLUMNS], test["label_cls"]
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, yhat)),
        "f1": float(f1_score(yte, yhat, zero_division=0)),
        "precision": float(precision_score(yte, yhat, zero_division=0)),
        "recall": float(recall_score(yte, yhat, zero_division=0)),
    }
    # Return probability for the latest row
    row = d[FEATURE_COLUMNS].iloc[[-1]]
    prob = float(model.predict_proba(row)[0, 1])
    label = int(prob >= 0.5)
    return ClassificationResult(metrics=metrics, probs=prob, label=label)
