"""
Tests for evaluation metrics and best-model selection. Named
test_model_loading.py per the required test matrix, but also covers
evaluation since the two are tightly coupled (selection depends on
metrics computed at "load"/evaluation time).
"""
import numpy as np

from app.models.evaluation import top_k_accuracy, is_emergency_condition, evaluate_predictions
from app.models.model_selector import select_best_model, composite_score


def test_is_emergency_condition_matches_keywords():
    assert is_emergency_condition("Call 999 or go to A&E immediately.")
    assert not is_emergency_condition("Rest and drink plenty of fluids.")


def test_top_k_accuracy_perfect_predictions():
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_proba = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    assert top_k_accuracy(y_true, y_proba, k=1) == 1.0


def test_evaluate_predictions_returns_expected_keys():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.6]])
    metrics = evaluate_predictions(y_true, y_proba, ["Asthma", "Cold"], [True, False])
    d = metrics.to_dict()
    for key in ["precision_micro", "recall_micro", "f1_micro", "top_1_accuracy",
                "top_3_accuracy", "top_5_accuracy", "emergency_recall"]:
        assert key in d


def test_select_best_model_prefers_higher_composite_score():
    docs = [
        {"name": "a", "type": "rnn", "metrics": {"top_3_accuracy": 0.4, "emergency_recall": 0.3, "top_1_accuracy": 0.2, "f1_micro": 0.2}},
        {"name": "b", "type": "bert", "metrics": {"top_3_accuracy": 0.9, "emergency_recall": 0.8, "top_1_accuracy": 0.5, "f1_micro": 0.5}},
    ]
    best = select_best_model(docs)
    assert best["name"] == "b"
    assert composite_score(docs[1]["metrics"]) > composite_score(docs[0]["metrics"])


def test_select_best_model_handles_empty_list():
    assert select_best_model([]) is None
