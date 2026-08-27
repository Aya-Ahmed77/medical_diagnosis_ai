"""
Training orchestration service.

Coordinates: load conditions from MongoDB -> build multi-label dataset ->
fit preprocessing pipeline -> train each requested model -> evaluate ->
persist model file to GridFS + metadata to the Models collection.

Each train_* function is independent and catches its own exceptions so
that, e.g., a missing GPU/library for the transformer doesn't prevent the
baseline/RNN/LSTM from training and being reported.
"""
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from app.config import get_config
from app.database.schemas import ConditionsRepository, ModelsRepository, ModelDocument
from app.database import gridfs_store
from app.preprocessing.dataset_builder import build_multilabel_dataset, train_val_test_split, DatasetRow
from app.preprocessing.pipeline import PreprocessingPipeline, save_dataset_meta
from app.models.evaluation import evaluate_predictions, is_emergency_condition
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _emergency_flags(label_names: List[str], conditions_by_name: Dict[str, dict]) -> List[bool]:
    flags = []
    for name in label_names:
        cond = conditions_by_name.get(name, {})
        flags.append(is_emergency_condition(cond.get("warnings", "")))
    return flags


def prepare_dataset():
    """Phase: dataset preparation. Returns (train, val, test, label_space,
    pipeline, conditions_by_name)."""
    cfg = get_config()
    repo = ConditionsRepository()
    conditions = repo.list_all()
    if not conditions:
        raise RuntimeError(
            "No conditions found in MongoDB. Run the scraper first "
            "(POST /api/data/scrape or scripts/run_scraper.py)."
        )

    rows, label_space = build_multilabel_dataset(conditions)
    train, val, test = train_val_test_split(rows)

    pipeline = PreprocessingPipeline().fit(rows)
    pipeline.save()
    save_dataset_meta(rows, label_space)

    conditions_by_name = {c["condition"]: c for c in conditions}
    logger.info(
        "Dataset ready: train=%d val=%d test=%d labels=%d",
        len(train), len(val), len(test), len(label_space),
    )
    return train, val, test, label_space, pipeline, conditions_by_name


def _save_model_record(name: str, model_type: str, local_artifact_path: str,
                        labels: List[str], metrics: dict, is_directory: bool = False) -> str:
    if is_directory:
        gridfs_id = gridfs_store.store_directory_as_zip(local_artifact_path, f"{name}.zip")
    else:
        gridfs_id = gridfs_store.store_file(local_artifact_path, filename=os.path.basename(local_artifact_path))

    doc = ModelDocument(
        name=name, type=model_type, gridfs_id=gridfs_id, labels=labels,
        metrics=metrics, created=datetime.now(timezone.utc).isoformat(),
    )
    ModelsRepository().save(doc.to_dict())
    logger.info("Persisted model '%s' (type=%s) metadata + GridFS artifact.", name, model_type)
    return gridfs_id


def train_baseline(train: List[DatasetRow], val: List[DatasetRow], test: List[DatasetRow],
                    label_space: List[str], pipeline: PreprocessingPipeline,
                    conditions_by_name: Dict[str, dict]) -> dict:
    from app.models.baseline_tfidf import TfidfLogRegModel

    cfg = get_config()
    y_train = pipeline.transform_labels(train)
    y_test = pipeline.transform_labels(test)

    model = TfidfLogRegModel().fit(pipeline.texts(train), y_train)
    y_proba = model.predict_proba(pipeline.texts(test))

    flags = _emergency_flags(pipeline.classes, conditions_by_name)
    metrics = evaluate_predictions(y_test, y_proba, pipeline.classes, flags).to_dict()

    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = model.save(tmp)
        _save_model_record("tfidf_logreg_v1", "tfidf_logreg", bundle_path, pipeline.classes, metrics)

    return {"type": "tfidf_logreg", "metrics": metrics}


def train_rnn(train, val, test, label_space, pipeline, conditions_by_name) -> dict:
    try:
        import tensorflow as tf  # noqa: F401
        from app.models.sequence_utils import SequenceVectorizer
        from app.models.rnn_model import build_rnn_model, train_rnn_model, predict_proba
    except ImportError as exc:
        logger.error("TensorFlow not available -- skipping RNN training: %s", exc)
        return {"type": "rnn", "error": "tensorflow_not_available", "metrics": None}

    cfg = get_config()
    vectorizer = SequenceVectorizer().fit(pipeline.texts(train))
    X_train = vectorizer.transform(pipeline.texts(train))
    X_val = vectorizer.transform(pipeline.texts(val))
    X_test = vectorizer.transform(pipeline.texts(test))

    y_train = pipeline.transform_labels(train)
    y_val = pipeline.transform_labels(val)
    y_test = pipeline.transform_labels(test)

    model = build_rnn_model(vocab_size=cfg.VOCAB_SIZE, num_labels=len(pipeline.classes))

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "rnn_model.keras")
        train_rnn_model(model, X_train, y_train, X_val, y_val, ckpt)
        y_proba = predict_proba(model, X_test)

        flags = _emergency_flags(pipeline.classes, conditions_by_name)
        metrics = evaluate_predictions(y_test, y_proba, pipeline.classes, flags).to_dict()

        vectorizer.save(tmp)
        _save_model_record("rnn_v1", "rnn", ckpt, pipeline.classes, metrics)
        # tokenizer stored alongside as a second small artifact
        _save_model_record("rnn_v1_tokenizer", "rnn_tokenizer",
                            os.path.join(tmp, "keras_tokenizer.joblib"), pipeline.classes, metrics)

    return {"type": "rnn", "metrics": metrics}


def train_lstm(train, val, test, label_space, pipeline, conditions_by_name) -> dict:
    try:
        import tensorflow as tf  # noqa: F401
        from app.models.sequence_utils import SequenceVectorizer
        from app.models.lstm_model import build_lstm_model, train_lstm_model, predict_proba
    except ImportError as exc:
        logger.error("TensorFlow not available -- skipping LSTM training: %s", exc)
        return {"type": "lstm", "error": "tensorflow_not_available", "metrics": None}

    cfg = get_config()
    vectorizer = SequenceVectorizer().fit(pipeline.texts(train))
    X_train = vectorizer.transform(pipeline.texts(train))
    X_val = vectorizer.transform(pipeline.texts(val))
    X_test = vectorizer.transform(pipeline.texts(test))

    y_train = pipeline.transform_labels(train)
    y_val = pipeline.transform_labels(val)
    y_test = pipeline.transform_labels(test)

    model = build_lstm_model(vocab_size=cfg.VOCAB_SIZE, num_labels=len(pipeline.classes))

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "lstm_model.keras")
        train_lstm_model(model, X_train, y_train, X_val, y_val, ckpt)
        y_proba = predict_proba(model, X_test)

        flags = _emergency_flags(pipeline.classes, conditions_by_name)
        metrics = evaluate_predictions(y_test, y_proba, pipeline.classes, flags).to_dict()

        vectorizer.save(tmp)
        _save_model_record("lstm_v1", "lstm", ckpt, pipeline.classes, metrics)
        _save_model_record("lstm_v1_tokenizer", "lstm_tokenizer",
                            os.path.join(tmp, "keras_tokenizer.joblib"), pipeline.classes, metrics)

    return {"type": "lstm", "metrics": metrics}


def train_transformer(train, val, test, label_space, pipeline, conditions_by_name) -> dict:
    try:
        import torch  # noqa: F401
        from app.models.transformer_model import (
            load_tokenizer_and_model, fine_tune_transformer, predict_proba, save_transformer,
        )
    except ImportError as exc:
        logger.error("PyTorch/Transformers not available -- skipping transformer training: %s", exc)
        return {"type": "transformer", "error": "torch_transformers_not_available", "metrics": None}

    y_train = pipeline.transform_labels(train)
    y_val = pipeline.transform_labels(val)
    y_test = pipeline.transform_labels(test)

    try:
        tokenizer, model, resolved_name = load_tokenizer_and_model(num_labels=len(pipeline.classes))
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load any transformer checkpoint (offline?): %s", exc)
        return {"type": "transformer", "error": f"model_download_failed: {exc}", "metrics": None}

    with tempfile.TemporaryDirectory() as tmp:
        try:
            fine_tune_transformer(
                tokenizer, model, pipeline.texts(train), y_train,
                pipeline.texts(val), y_val, output_dir=tmp,
            )
            y_proba = predict_proba(tokenizer, model, pipeline.texts(test))
        except Exception as exc:  # noqa: BLE001 -- e.g. OOM on CPU-only box
            logger.error("Transformer fine-tuning failed: %s", exc)
            return {"type": "transformer", "error": str(exc), "metrics": None}

        flags = _emergency_flags(pipeline.classes, conditions_by_name)
        metrics = evaluate_predictions(y_test, y_proba, pipeline.classes, flags).to_dict()
        metrics["base_checkpoint"] = resolved_name

        save_dir = os.path.join(tmp, "saved_model")
        save_transformer(tokenizer, model, save_dir)
        _save_model_record("transformer_v1", "transformer", save_dir, pipeline.classes, metrics, is_directory=True)

    return {"type": "transformer", "metrics": metrics}


def train_all_models(model_types: Optional[List[str]] = None) -> dict:
    """Run the full training phase for the requested model types (default:
    all four). Returns a summary dict per model, including any that were
    skipped due to missing dependencies/infra."""
    model_types = model_types or ["tfidf_logreg", "rnn", "lstm", "transformer"]
    train, val, test, label_space, pipeline, conditions_by_name = prepare_dataset()

    results = {}
    dispatch = {
        "tfidf_logreg": train_baseline,
        "rnn": train_rnn,
        "lstm": train_lstm,
        "transformer": train_transformer,
    }
    for mtype in model_types:
        fn = dispatch.get(mtype)
        if fn is None:
            results[mtype] = {"error": "unknown_model_type"}
            continue
        try:
            results[mtype] = fn(train, val, test, label_space, pipeline, conditions_by_name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training failed for %s", mtype)
            results[mtype] = {"type": mtype, "error": str(exc), "metrics": None}

    return {
        "dataset": {
            "train_size": len(train), "val_size": len(val), "test_size": len(test),
            "num_labels": len(label_space),
        },
        "models": results,
    }
