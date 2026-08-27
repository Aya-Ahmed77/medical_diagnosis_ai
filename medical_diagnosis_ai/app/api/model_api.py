"""
Model API (project requirement 14).

Endpoints:
  POST /api/model/train           -- train one/all models, evaluate, persist
  GET  /api/model/list             -- list all saved model metadata records
  GET  /api/model/best             -- return the currently selected best model
  POST /api/model/predict          -- run a prediction for free-text symptoms
"""
from flask import Blueprint, jsonify, request

from app.database.schemas import ModelsRepository
from app.models.model_selector import select_best_model, explain_selection
from app.services.training_service import train_all_models
from app.services.prediction_service import predict_conditions, get_best_model_document, ModelNotAvailableError
from app.utils.logger import get_logger
from app.utils.validators import ValidationError, require_fields, optional_int

logger = get_logger(__name__)
model_api = Blueprint("model_api", __name__, url_prefix="/api/model")


@model_api.errorhandler(ValidationError)
def _handle_validation_error(err):
    return jsonify({"error": str(err)}), 400


@model_api.route("/train", methods=["POST"])
def train():
    """Body (optional): { "model_types": ["tfidf_logreg", "rnn", "lstm", "transformer"] }"""
    payload = request.get_json(silent=True) or {}
    model_types = payload.get("model_types")
    try:
        results = train_all_models(model_types=model_types)
        return jsonify({"status": "completed", **results}), 200
    except RuntimeError as exc:
        return jsonify({"status": "failed", "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training endpoint failed")
        return jsonify({"status": "failed", "error": str(exc)}), 500


@model_api.route("/list", methods=["GET"])
def list_models():
    docs = ModelsRepository().list_all()
    for d in docs:
        d["_id"] = str(d["_id"])
    return jsonify({"count": len(docs), "models": docs}), 200


@model_api.route("/best", methods=["GET"])
def best_model():
    docs = ModelsRepository().list_all()
    docs_with_metrics = [d for d in docs if d.get("metrics")]
    best = select_best_model(docs_with_metrics)
    if not best:
        return jsonify({"error": "No trained models with metrics available yet."}), 404
    best["_id"] = str(best["_id"])
    explanation = explain_selection(best, docs_with_metrics)
    return jsonify({"best_model": best, "explanation": explanation}), 200


@model_api.route("/predict", methods=["POST"])
def predict():
    """Body: { "symptoms_text": "...", "age": 34, "gender": "female", "top_n": 5 }"""
    payload = request.get_json(silent=True) or {}
    require_fields(payload, ["symptoms_text"])

    ok_age, age = optional_int(payload, "age")
    if not ok_age:
        raise ValidationError("'age' must be an integer.")
    ok_topn, top_n = optional_int(payload, "top_n")
    if not ok_topn:
        raise ValidationError("'top_n' must be an integer.")

    gender = payload.get("gender")

    try:
        result = predict_conditions(
            free_text=payload["symptoms_text"], age=age, gender=gender, top_n=top_n,
        )
        return jsonify(result), 200
    except ModelNotAvailableError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500
