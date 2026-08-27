"""
Flask API endpoint tests. External calls (MongoDB, scraping, model
training/inference) are mocked so these run fast and offline, exercising
request validation, routing, and response shape/status codes.
"""
import pytest

from app import create_app
from app.api import data_api, preprocessing_api, model_api


@pytest.fixture
def client(monkeypatch):
    # Patch the names as imported into app/__init__.py (not the source module),
    # since `from x import y` binds a new local reference to `y`.
    monkeypatch.setattr("app.ensure_indexes", lambda: None)
    monkeypatch.setattr("app.check_connection", lambda: True)
    app = create_app()
    app.testing = True
    return app.test_client()


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_scrape_endpoint_success(client, monkeypatch):
    monkeypatch.setattr(data_api, "run_scrape", lambda limit, force_refresh: {
        "total_links_found": 10, "scraped": 5, "skipped_existing": 3, "failed": 2, "failed_urls": [],
    })
    resp = client.post("/api/data/scrape", json={"limit": 5})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["scraped"] == 5


def test_scrape_endpoint_bad_limit_type(client):
    resp = client.post("/api/data/scrape", json={"limit": "not-an-int"})
    assert resp.status_code == 400


def test_scrape_endpoint_connection_error(client, monkeypatch):
    def _raise(*a, **k):
        raise ConnectionError("cannot reach nhsinform.scot")
    monkeypatch.setattr(data_api, "run_scrape", _raise)
    resp = client.post("/api/data/scrape", json={})
    assert resp.status_code == 502


def test_list_conditions_endpoint(client, monkeypatch):
    class _FakeRepo:
        def list_all(self, limit=0):
            return [{"_id": "x1", "condition": "Asthma", "symptoms": [], "causes": [],
                      "warnings": "w", "recommendations": "r"}]
    monkeypatch.setattr(data_api, "ConditionsRepository", _FakeRepo)
    resp = client.get("/api/data/conditions")
    assert resp.status_code == 200
    assert resp.get_json()["count"] == 1


def test_predict_endpoint_missing_field(client):
    resp = client.post("/api/model/predict", json={})
    assert resp.status_code == 400


def test_predict_endpoint_success(client, monkeypatch):
    monkeypatch.setattr(model_api, "predict_conditions", lambda **kwargs: {
        "predictions": [{"condition": "Asthma", "probability": 0.9, "warnings": "w", "recommendations": "r"}],
        "disclaimer": "not a diagnosis",
        "model_used": {"name": "tfidf_logreg_v1", "type": "tfidf_logreg"},
        "input": kwargs,
    })
    resp = client.post("/api/model/predict", json={"symptoms_text": "sore throat", "age": 30})
    assert resp.status_code == 200
    assert resp.get_json()["predictions"][0]["condition"] == "Asthma"


def test_train_endpoint_success(client, monkeypatch):
    monkeypatch.setattr(model_api, "train_all_models", lambda model_types=None: {
        "dataset": {"train_size": 10, "val_size": 2, "test_size": 2, "num_labels": 3},
        "models": {"tfidf_logreg": {"type": "tfidf_logreg", "metrics": {"top_3_accuracy": 0.6}}},
    })
    resp = client.post("/api/model/train", json={"model_types": ["tfidf_logreg"]})
    assert resp.status_code == 200
    assert "tfidf_logreg" in resp.get_json()["models"]
