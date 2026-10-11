from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_predict():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert response.json()["predicted_class"] in ["setosa", "versicolor", "virginica"]
