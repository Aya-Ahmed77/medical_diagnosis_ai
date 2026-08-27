"""
The reusable preprocessing pipeline object.

This wraps text_cleaning + a fitted MultiLabelBinarizer (the label
encoder for the multi-label target) into one object that is:
  * fit once during dataset preparation,
  * saved to disk as an artifact,
  * loaded again unchanged for every model's training AND for live
    inference, guaranteeing training/inference consistency
    (project requirement 6).
"""
import json
import os
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from app.config import get_config
from app.preprocessing.dataset_builder import DatasetRow
from app.preprocessing.text_cleaning import build_model_input_text, split_free_text_into_clauses
from app.utils.logger import get_logger

logger = get_logger(__name__)

ARTIFACT_FILENAME = "label_binarizer.joblib"
DATASET_META_FILENAME = "dataset_meta.json"


class PreprocessingPipeline:
    def __init__(self, label_space: List[str] = None):
        self.label_binarizer = MultiLabelBinarizer(classes=label_space) if label_space else MultiLabelBinarizer()
        self._fitted = label_space is not None

    def fit(self, rows: List[DatasetRow]) -> "PreprocessingPipeline":
        all_labels = [row.labels for row in rows]
        self.label_binarizer.fit(all_labels)
        self._fitted = True
        return self

    def transform_labels(self, rows: List[DatasetRow]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PreprocessingPipeline must be fit() before transform_labels().")
        return self.label_binarizer.transform([row.labels for row in rows])

    def texts(self, rows: List[DatasetRow]) -> List[str]:
        return [row.text for row in rows]

    @property
    def classes(self) -> List[str]:
        return list(self.label_binarizer.classes_)

    def prepare_inference_text(self, free_text: str, age: int = None, gender: str = None) -> str:
        """Turn raw user input into the SAME text representation used at
        training time. Age/gender are accepted for API completeness (per
        requirement 13) but are metadata, not folded into the text model
        input, to avoid the text classifier learning spurious shortcuts
        from small demographic fields.
        """
        clauses = split_free_text_into_clauses(free_text)
        return build_model_input_text(clauses)

    def save(self, artifacts_dir: str = None) -> str:
        cfg = get_config()
        artifacts_dir = artifacts_dir or cfg.ARTIFACTS_DIR
        os.makedirs(artifacts_dir, exist_ok=True)
        path = os.path.join(artifacts_dir, ARTIFACT_FILENAME)
        joblib.dump(self.label_binarizer, path)
        logger.info("Saved preprocessing pipeline (label binarizer) to %s", path)
        return path

    @classmethod
    def load(cls, artifacts_dir: str = None) -> "PreprocessingPipeline":
        cfg = get_config()
        artifacts_dir = artifacts_dir or cfg.ARTIFACTS_DIR
        path = os.path.join(artifacts_dir, ARTIFACT_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved preprocessing pipeline at {path}. Run dataset "
                "preparation / training first."
            )
        pipeline = cls()
        pipeline.label_binarizer = joblib.load(path)
        pipeline._fitted = True
        return pipeline


def save_dataset_meta(rows: List[DatasetRow], label_space: List[str], artifacts_dir: str = None) -> str:
    """Persist a small JSON summary of the built dataset (counts, real vs
    synthetic split, label space) for the experimentation report and for
    quick sanity checks without re-running the whole pipeline."""
    cfg = get_config()
    artifacts_dir = artifacts_dir or cfg.ARTIFACTS_DIR
    os.makedirs(artifacts_dir, exist_ok=True)
    meta = {
        "total_rows": len(rows),
        "real_rows": sum(1 for r in rows if not r.is_synthetic),
        "synthetic_rows": sum(1 for r in rows if r.is_synthetic),
        "num_labels": len(label_space),
        "labels": label_space,
    }
    path = os.path.join(artifacts_dir, DATASET_META_FILENAME)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return path
