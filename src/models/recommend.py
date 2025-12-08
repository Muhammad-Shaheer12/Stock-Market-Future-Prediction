from __future__ import annotations
from typing import Dict
import pandas as pd
from src.models.train import HORIZONS
from src.features.engineering import FEATURE_COLUMNS
import joblib
from pathlib import Path
from src.models.train import load_production_model_dir


def recommend(df_by_symbol: Dict[str, pd.DataFrame], top_k: int = 5, horizon: int = 7):
    prod = load_production_model_dir()
    if not prod:
        raise RuntimeError("No production model available")
    if horizon not in HORIZONS:
        raise ValueError("Unsupported horizon")
    model = joblib.load(Path(prod) / f"model_h{horizon}.joblib")
    scores = []
    for sym, df in df_by_symbol.items():
        X = df[FEATURE_COLUMNS].fillna(0.0).iloc[-1:]
        pred = float(model.predict(X)[0])
        scores.append((sym, pred))
    # diversify slightly by penalizing correlation via recent returns variance proxy
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return [{"symbol": s, "score": p} for s, p in ranked[:top_k]]
