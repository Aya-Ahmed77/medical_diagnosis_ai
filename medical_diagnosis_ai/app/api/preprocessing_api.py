"""
Preprocessing API (project requirement 14).

Endpoints:
  POST /api/preprocessing/prepare-dataset -- clean scraped data, build the
                                              multi-label dataset, fit and
                                              save the preprocessing pipeline
  GET  /api/preprocessing/stats            -- return stats about the last
                                              prepared dataset (from the
                                              saved dataset_meta.json)
"""
import json
import os

from flask import Blueprint, jsonify

from app.config import get_config
from app.services.training_service import prepare_dataset
from app.preprocessing.pipeline import DATASET_META_FILENAME
from app.utils.logger import get_logger

logger = get_logger(__name__)
preprocessing_api = Blueprint("preprocessing_api", __name__, url_prefix="/api/preprocessing")


@preprocessing_api.route("/prepare-dataset", methods=["POST"])
def prepare_dataset_endpoint():
    try:
        train, val, test, label_space, pipeline, _ = prepare_dataset()
        return jsonify({
            "status": "completed",
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "num_labels": len(label_space),
            "labels": label_space,
        }), 200
    except RuntimeError as exc:
        return jsonify({"status": "failed", "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Preprocessing failed")
        return jsonify({"status": "failed", "error": str(exc)}), 500


@preprocessing_api.route("/stats", methods=["GET"])
def dataset_stats():
    cfg = get_config()
    path = os.path.join(cfg.ARTIFACTS_DIR, DATASET_META_FILENAME)
    if not os.path.exists(path):
        return jsonify({"error": "No dataset has been prepared yet."}), 404
    with open(path) as fh:
        meta = json.load(fh)
    return jsonify(meta), 200
