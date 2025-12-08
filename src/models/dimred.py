from __future__ import annotations
import pandas as pd
from sklearn.decomposition import PCA
from src.features.engineering import FEATURE_COLUMNS


def pca_2d(df: pd.DataFrame, n_components: int = 2):
    d = df[FEATURE_COLUMNS].dropna()
    if len(d) < 10:
        raise ValueError("Insufficient data for PCA")
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(d)
    evr = pca.explained_variance_ratio_.tolist()
    # Use last N dates to match points
    labels = df.loc[d.index, "date"].dt.strftime("%Y-%m-%d").tolist()
    points = [[float(x), float(y)] for x, y in proj]
    return {
        "explained_variance_ratio": [float(x) for x in evr],
        "labels": labels,
        "points": points,
    }
