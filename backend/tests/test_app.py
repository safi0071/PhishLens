import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app

def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"

def test_predict_rejects_invalid_payload():
    client = app.test_client()
    response = client.post("/predict", json={"wrong": "shape"})
    # If the model is missing, endpoint reports that first; both are safe error responses.
    assert response.status_code in (400, 503)
