from fastapi.testclient import TestClient

from wb_sentiment import api


class FixedModel:
    def predict(self, texts):
        assert texts == ["Отличное качество удобно"]
        return ["positive"]


def test_health_reports_model_ready(monkeypatch):
    monkeypatch.setattr(api, "get_bundle", lambda: {"model_version": "1.0.0"})
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["model_ready"] is True


def test_predict_returns_version_and_sentiment(monkeypatch):
    monkeypatch.setattr(api, "get_bundle", lambda: {
        "model": FixedModel(), "model_version": "1.0.0", "model_name": "fixed-demo"
    })
    response = TestClient(api.app).post(
        "/predict", json={"text": "Отличное качество", "pros": "удобно"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "sentiment": "positive", "model_version": "1.0.0", "model_id": "fixed-demo"
    }


def test_predict_rejects_empty_text():
    response = TestClient(api.app).post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_health_returns_controlled_error_when_model_missing(monkeypatch):
    def missing_model():
        raise FileNotFoundError("not found")

    monkeypatch.setattr(api, "get_bundle", missing_model)
    response = TestClient(api.app).get("/health")
    assert response.status_code == 503
    assert response.json()["detail"].startswith("Модель не готова")


def test_predict_returns_controlled_error_when_model_missing(monkeypatch):
    def missing_model():
        raise FileNotFoundError("not found")

    monkeypatch.setattr(api, "get_bundle", missing_model)
    response = TestClient(api.app).post("/predict", json={"text": "пример отзыва"})
    assert response.status_code == 503
    assert response.json()["detail"].startswith("Модель не готова")
