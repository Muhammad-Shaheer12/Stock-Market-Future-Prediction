from __future__ import annotations
import pandas as pd
from sklearn.cluster import KMeans
from src.features.engineering import FEATURE_COLUMNS


def kmeans_clusters(df: pd.DataFrame, k: int = 4):
    d = df[FEATURE_COLUMNS].dropna()
    if len(d) < k:
        raise ValueError("Insufficient data for clustering")
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(d)
    centers = km.cluster_centers_.tolist()
    dates = df.loc[d.index, "date"].dt.strftime("%Y-%m-%d").tolist()
    return {"k": k, "labels": labels.tolist(), "centers": centers, "dates": dates}
