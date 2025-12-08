from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
from src.features.engineering import FEATURE_COLUMNS
from src.utils.config import MODELS_DIR, HORIZONS

HORIZONS = HORIZONS


def walk_forward_split(df: pd.DataFrame, train_ratio=0.8):
    n = len(df)
    split = int(n * train_ratio)
    return df.iloc[:split], df.iloc[split:]


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_models(df: pd.DataFrame) -> Path:
    X = df[FEATURE_COLUMNS].fillna(0.0)
    results = {}
    models = {}
    for h in HORIZONS:
        y = df[f"label_{h}"].fillna(0.0)
        d = pd.concat([X, y], axis=1).dropna()
        Xh, yh = d[FEATURE_COLUMNS], d[f"label_{h}"]
        train, test = walk_forward_split(pd.concat([Xh, yh], axis=1))
        Xtr, ytr = train[FEATURE_COLUMNS], train[f"label_{h}"]
        Xte, yte = test[FEATURE_COLUMNS], test[f"label_{h}"]
        # Baseline
        lr = LinearRegression().fit(Xtr, ytr)
        lr_pred = lr.predict(Xte)
        lr_m = metrics(yte, lr_pred)
        # Main model
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        xgb.fit(Xtr, ytr)
        xgb_pred = xgb.predict(Xte)
        xgb_m = metrics(yte, xgb_pred)
        if xgb_m["rmse"] <= lr_m["rmse"]:
            models[h] = xgb
            results[h] = {"winner": "xgb", "baseline": lr_m, "model": xgb_m}
        else:
            models[h] = lr
            results[h] = {"winner": "lr", "baseline": lr_m, "model": xgb_m}
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_dir = MODELS_DIR / f"mh_price_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for h, model in models.items():
        joblib.dump(model, out_dir / f"model_h{h}.joblib")
    meta = {
        "timestamp": ts,
        "horizons": HORIZONS,
        "features": FEATURE_COLUMNS,
        "results": results,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    # Update production pointer
    with open(MODELS_DIR / "production.txt", "w") as f:
        f.write(str(out_dir))
    return out_dir


def load_production_model_dir() -> Path | None:
    p = MODELS_DIR / "production.txt"
    if p.exists():
        path = Path(p.read_text().strip())
        if path.exists():
            return path
    # fallback: latest
    candidates = [d for d in MODELS_DIR.glob("mh_price_*") if d.is_dir()]
    return max(candidates, key=lambda x: x.stat().st_mtime) if candidates else None
