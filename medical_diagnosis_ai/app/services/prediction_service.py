"""
Prediction service (project requirement 13).

Loads the currently-selected best model (per app.models.model_selector),
runs a free-text symptom description through it, ranks conditions by
probability, and enriches the top results with warnings/recommendations
pulled live from the Conditions collection.

Predictions are always returned alongside cfg.MEDICAL_DISCLAIMER and are
explicitly labeled as model-generated possibilities, never as a diagnosis.
"""
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np

from app.config import get_config
from app.database.schemas import ConditionsRepository, ModelsRepository
from app.database import gridfs_store
from app.models.model_selector import select_best_model
from app.preprocessing.pipeline import PreprocessingPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelNotAvailableError(RuntimeError):
    pass


def _load_model_for_inference(model_doc: dict):
    """Load the appropriate model object based on its stored `type`."""
    mtype = model_doc["type"]
    gridfs_id = model_doc["gridfs_id"]

    if mtype == "tfidf_logreg":
        from app.models.baseline_tfidf import TfidfLogRegModel
        with tempfile.TemporaryDirectory() as tmp:
            local_path = os.path.join(tmp, "bundle.joblib")
            gridfs_store.load_to_file(gridfs_id, local_path)
            model = TfidfLogRegModel.load(local_path)
        return ("sklearn", model)

    if mtype in ("rnn", "lstm"):
        import tensorflow as tf
        from app.models.sequence_utils import SequenceVectorizer

        tokenizer_doc = ModelsRepository().get_by_name(f"{model_doc['name']}_tokenizer")
        if not tokenizer_doc:
            raise ModelNotAvailableError(f"Tokenizer record for '{model_doc['name']}' not found.")

        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.keras")
            gridfs_store.load_to_file(gridfs_id, model_path)
            keras_model = tf.keras.models.load_model(model_path)

            gridfs_store.load_to_file(tokenizer_doc["gridfs_id"], os.path.join(tmp, "keras_tokenizer.joblib"))
            vectorizer = SequenceVectorizer.load(tmp)
        return ("keras", (keras_model, vectorizer))

    if mtype == "transformer":
        from app.models.transformer_model import load_transformer
        with tempfile.TemporaryDirectory() as tmp:
            gridfs_store.load_zip_to_directory(gridfs_id, tmp)
            tokenizer, model = load_transformer(tmp, num_labels=len(model_doc["labels"]))
        return ("transformer", (tokenizer, model))

    raise ModelNotAvailableError(f"Unknown model type '{mtype}'.")


def get_best_model_document() -> dict:
    docs = ModelsRepository().list_all()
    docs_with_metrics = [d for d in docs if d.get("metrics")]
    best = select_best_model(docs_with_metrics)
    if not best:
        raise ModelNotAvailableError(
            "No trained model with evaluation metrics is available yet. Train models first."
        )
    return best


def _predict_proba_for_kind(kind: str, payload, texts: List[str]) -> np.ndarray:
    if kind == "sklearn":
        return payload.predict_proba(texts)
    if kind == "keras":
        keras_model, vectorizer = payload
        X = vectorizer.transform(texts)
        return keras_model.predict(X, verbose=0)
    if kind == "transformer":
        from app.models.transformer_model import predict_proba
        tokenizer, model = payload
        return predict_proba(tokenizer, model, texts)
    raise ModelNotAvailableError(f"Unsupported model kind '{kind}'.")


def predict_conditions(
    free_text: str, age: Optional[int] = None, gender: Optional[str] = None,
    top_n: Optional[int] = None, model_doc: Optional[dict] = None,
) -> Dict:
    cfg = get_config()
    top_n = top_n or cfg.PREDICTION_TOP_N

    if not free_text or not free_text.strip():
        raise ValueError("free_text symptom description must not be empty.")

    model_doc = model_doc or get_best_model_document()
    pipeline = PreprocessingPipeline.load()
    kind, payload = _load_model_for_inference(model_doc)

    input_text = pipeline.prepare_inference_text(free_text, age=age, gender=gender)
    proba = _predict_proba_for_kind(kind, payload, [input_text])[0]

    labels = model_doc["labels"]
    ranked_idx = np.argsort(-proba)[:top_n]

    conditions_repo = ConditionsRepository()
    results = []
    for idx in ranked_idx:
        condition_name = labels[idx]
        doc = conditions_repo.get_by_name(condition_name) or {}
        results.append({
            "condition": condition_name,
            "probability": round(float(proba[idx]), 4),
            "warnings": doc.get("warnings", "No warning information available."),
            "recommendations": doc.get("recommendations", "No recommendation information available."),
        })

    return {
        "input": {"free_text": free_text, "age": age, "gender": gender},
        "model_used": {"name": model_doc["name"], "type": model_doc["type"]},
        "predictions": results,
        "disclaimer": cfg.MEDICAL_DISCLAIMER,
    }
