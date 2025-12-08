import pandas as pd
import numpy as np

from src.features.engineering import FEATURE_COLUMNS
from src.models.classify import train_and_predict
from src.models.dimred import pca_2d
from src.models.cluster import kmeans_clusters
from src.models.recommend import recommend as reco


def make_df(n=120):
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices + np.random.randn(n),
            "high": prices + np.abs(np.random.randn(n)),
            "low": prices - np.abs(np.random.randn(n)),
            "close": prices + np.random.randn(n),
            "adj_close": prices,
            "volume": np.random.randint(1_000_000, 5_000_000, size=n),
        }
    )
    # compute minimal features to satisfy FEATURE_COLUMNS
    from src.features.engineering import compute_features

    df = compute_features(df)
    return df


def test_classify_model_runs():
    df = make_df()
    res = train_and_predict(df, horizon=1)
    assert 0.0 <= res.probs <= 1.0
    assert set(res.metrics.keys()) >= {"accuracy", "f1", "precision", "recall"}


def test_pca_runs():
    df = make_df()
    out = pca_2d(df)
    assert "explained_variance_ratio" in out
    assert len(out["points"]) > 10


def test_kmeans_runs():
    df = make_df()
    out = kmeans_clusters(df, k=3)
    assert out["k"] == 3
    assert len(out["labels"]) > 10


def test_recommend_scores():
    # Prepare fake model path by writing simple linear predictions; skip actual joblib by mocking
    df_by_sym = {"AAA": make_df(), "BBB": make_df(), "CCC": make_df()}
    # monkeypatch load_production_model_dir and joblib.load inside recommend
    from src.models import recommend as recmod

    class DummyModel:
        def predict(self, X):
            return np.array([X[FEATURE_COLUMNS].sum(axis=1).values[0] * 0.0])

    from pathlib import Path

    recmod.load_production_model_dir = lambda: Path(".")
    import joblib

    joblib.load = lambda p: DummyModel()
    res = reco(df_by_sym, top_k=2, horizon=1)
    assert len(res) == 2
