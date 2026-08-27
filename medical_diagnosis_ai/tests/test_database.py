"""
Database layer tests using mongomock, so they run without a real MongoDB
instance -- per requirement 18 ("mock external requests/services where
appropriate").
"""
import mongomock
import pytest

from app.database import schemas


@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    client = mongomock.MongoClient()
    db = client["test_medical_diagnosis_ai"]
    monkeypatch.setattr("app.database.schemas.get_db", lambda: db)
    return db


def test_conditions_repository_upsert_and_get():
    repo = schemas.ConditionsRepository()
    doc = {
        "condition": "Asthma", "symptoms": ["wheezing"], "causes": ["allergens"],
        "warnings": "Call 999 if severe.", "recommendations": "Use inhaler.",
    }
    repo.upsert(doc)
    fetched = repo.get_by_name("Asthma")
    assert fetched is not None
    assert fetched["symptoms"] == ["wheezing"]


def test_conditions_repository_upsert_is_idempotent():
    repo = schemas.ConditionsRepository()
    doc = {
        "condition": "Asthma", "symptoms": ["wheezing"], "causes": [],
        "warnings": "w", "recommendations": "r",
    }
    repo.upsert(doc)
    doc["symptoms"] = ["wheezing", "coughing"]
    repo.upsert(doc)
    assert repo.count() == 1
    assert repo.get_by_name("Asthma")["symptoms"] == ["wheezing", "coughing"]


def test_condition_validation_rejects_missing_fields():
    with pytest.raises(ValueError):
        schemas.validate_condition_doc({"condition": "Asthma"})


def test_models_repository_save_and_get_best_by_metric():
    repo = schemas.ModelsRepository()
    repo.save({
        "name": "model_a", "type": "rnn", "gridfs_id": "abc",
        "labels": ["Asthma"], "metrics": {"top_3_accuracy": 0.5}, "created": "now",
    })
    repo.save({
        "name": "model_b", "type": "lstm", "gridfs_id": "def",
        "labels": ["Asthma"], "metrics": {"top_3_accuracy": 0.8}, "created": "now",
    })
    best = repo.get_best_by_metric("metrics.top_3_accuracy")
    assert best["name"] == "model_b"
