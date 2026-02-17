from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_valid():
    r = client.post("/predict", json={"text":"삼성전자는 오늘 상승했다"})
    assert r.status_code == 200
    data = r.json()
    assert "probabilities" in data
    assert isinstance(data["probabilities"], list)

def test_predict_invalid():
    r = client.post("/predict", json={})
    assert r.status_code == 422
