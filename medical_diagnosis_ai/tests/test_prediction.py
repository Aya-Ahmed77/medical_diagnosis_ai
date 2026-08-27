"""
Prediction service tests. The actual trained model and MongoDB are
mocked out so this test verifies the RANKING / ENRICHMENT / DISCLAIMER
logic in isolation, without requiring a real trained model or database.
"""
import numpy as np
import pytest

from app.services import prediction_service as ps


class _FakeSklearnModel:
    """Always returns fixed, known probabilities so ranking is deterministic."""
    def predict_proba(self, texts):
        return np.array([[0.1, 0.9, 0.3]])  # labels: Asthma, Common cold, Migraine


class _FakePipeline:
    def prepare_inference_text(self, free_text, age=None, gender=None):
        return "sore throat runny nose"


@pytest.fixture
def fake_model_doc():
    return {
        "name": "tfidf_logreg_v1", "type": "tfidf_logreg", "gridfs_id": "fake-id",
        "labels": ["Asthma", "Common cold", "Migraine"],
        "metrics": {"top_3_accuracy": 0.7, "emergency_recall": 0.6},
    }


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, fake_model_doc):
    monkeypatch.setattr(ps, "get_best_model_document", lambda: fake_model_doc)
    monkeypatch.setattr(ps.PreprocessingPipeline, "load", staticmethod(lambda: _FakePipeline()))
    monkeypatch.setattr(ps, "_load_model_for_inference", lambda doc: ("sklearn", _FakeSklearnModel()))

    class _FakeConditionsRepo:
        def get_by_name(self, name):
            return {
                "condition": name,
                "warnings": "Call 999 immediately." if name == "Common cold" else "Rest and hydrate.",
                "recommendations": "See a doctor if it persists.",
            }
    monkeypatch.setattr(ps, "ConditionsRepository", _FakeConditionsRepo)


def test_predict_conditions_ranks_by_probability_descending():
    result = ps.predict_conditions("sore throat and runny nose", top_n=3)
    probs = [p["probability"] for p in result["predictions"]]
    assert probs == sorted(probs, reverse=True)
    assert result["predictions"][0]["condition"] == "Common cold"


def test_predict_conditions_includes_disclaimer_and_warnings():
    result = ps.predict_conditions("sore throat", top_n=2)
    assert "not" in result["disclaimer"].lower() or "NOT" in result["disclaimer"]
    assert all("warnings" in p and "recommendations" in p for p in result["predictions"])


def test_predict_conditions_rejects_empty_text():
    with pytest.raises(ValueError):
        ps.predict_conditions("   ")
