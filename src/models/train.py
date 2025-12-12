from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
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
        candidates = {}

        lr = LinearRegression().fit(Xtr, ytr)
        candidates["lr"] = metrics(yte, lr.predict(Xte))

        ridge = Ridge(alpha=1.0, random_state=42).fit(Xtr, ytr)
        candidates["ridge"] = metrics(yte, ridge.predict(Xte))

        enet = ElasticNet(
            alpha=0.001, l1_ratio=0.2, random_state=42, max_iter=5000
        ).fit(Xtr, ytr)
        candidates["elasticnet"] = metrics(yte, enet.predict(Xte))

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ).fit(Xtr, ytr)
        candidates["rf"] = metrics(yte, rf.predict(Xte))

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
        candidates["xgb"] = metrics(yte, xgb.predict(Xte))

        # Pick lowest RMSE
        winner = min(candidates.items(), key=lambda kv: kv[1]["rmse"])[0]
        model_by_name = {
            "lr": lr,
            "ridge": ridge,
            "elasticnet": enet,
            "rf": rf,
            "xgb": xgb,
        }
        models[h] = model_by_name[winner]
        results[h] = {
            "winner": winner,
            "metrics": candidates[winner],
            "candidates": candidates,
        }
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
