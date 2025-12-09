from fastapi.testclient import TestClient
import os

os.environ.setdefault("ALPHAVANTAGE_API_KEY", "dummy")
from src.api.app import app  # noqa: E402

client = TestClient(app)


def fake_js():
    # Generate ~120 daily entries to satisfy feature windows
    series = {}
    base_price = 100.0
    base_vol = 1_000_000
    for i in range(1, 121):
        day = f"2025-01-{i:02d}"
        # Skip invalid days >31 by rolling into Feb/Mar to keep keys unique
        if i > 31:
            month = 2 + (i - 32) // 30
            day_in_month = (i - 32) % 30 + 1
            day = f"2025-{month:02d}-{day_in_month:02d}"
        price = base_price + 0.5 * i
        series[day] = {
            "1. open": str(price - 0.5),
            "2. high": str(price + 0.5),
            "3. low": str(price - 1.0),
            "4. close": str(price),
            "5. adjusted close": str(price),
            "6. volume": str(base_vol + i * 1000),
        }
    return {"Time Series (Daily)": series}


def test_classify_endpoint(monkeypatch):
    # Patch API to use a dummy AV client
    import src.api.app as api_mod

    class DummyClient:
        def __init__(self):
            pass

        def daily_adjusted(self, symbol, outputsize="compact"):
            return fake_js()

    monkeypatch.setattr(api_mod, "AlphaVantageClient", DummyClient)
    # Monkeypatch classification to avoid data-size constraints
    from src.models import classify as cls

    class DummyRes:
        metrics = {"acc": 1.0}
        probs = 0.75
        label = 1

    monkeypatch.setattr(cls, "train_and_predict", lambda df, horizon=1: DummyRes())
    r = client.post("/classify/movement", json={"symbol": "TEST", "horizon": 1})
    assert r.status_code == 200
    js = r.json()
    assert "metrics" in js and "prob_up" in js


def test_pca_endpoint(monkeypatch):
    import src.api.app as api_mod

    class DummyClient:
        def __init__(self):
            pass

        def daily_adjusted(self, symbol, outputsize="compact"):
            return fake_js()

    monkeypatch.setattr(api_mod, "AlphaVantageClient", DummyClient)
    r = client.get("/dimred/pca?symbol=TEST")
    assert r.status_code == 200
    js = r.json()
    assert "explained_variance_ratio" in js and "points" in js


def test_cluster_endpoint(monkeypatch):
    import src.api.app as api_mod

    class DummyClient:
        def __init__(self):
            pass

        def daily_adjusted(self, symbol, outputsize="compact"):
            return fake_js()

    monkeypatch.setattr(api_mod, "AlphaVantageClient", DummyClient)
    r = client.get("/cluster/assets?symbol=TEST&k=2")
    assert r.status_code == 200
    js = r.json()
    assert js["k"] == 2
