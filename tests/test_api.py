from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()
