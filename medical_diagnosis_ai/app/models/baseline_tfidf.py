"""
Baseline model: TF-IDF vectorization + Logistic Regression, wrapped in
OneVsRestClassifier for multi-label output (per-condition probabilities).
"""
import os
from typing import List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from app.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_TYPE = "tfidf_logreg"
VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
CLASSIFIER_FILENAME = "tfidf_logreg_classifier.joblib"


class TfidfLogRegModel:
    def __init__(self):
        cfg = get_config()
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.TFIDF_MAX_FEATURES,
            ngram_range=(1, cfg.TFIDF_NGRAM_MAX),
        )
        self.classifier = OneVsRestClassifier(
            LogisticRegression(max_iter=cfg.LOGREG_MAX_ITER)
        )
        self._fitted = False

    def fit(self, texts: List[str], y: np.ndarray) -> "TfidfLogRegModel":
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fit() before predict_proba().")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def save(self, local_dir: str) -> str:
        """Save both vectorizer and classifier; return a single bundled
        joblib path (simplifies GridFS storage as one artifact)."""
        os.makedirs(local_dir, exist_ok=True)
        bundle_path = os.path.join(local_dir, "tfidf_logreg_bundle.joblib")
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, bundle_path)
        logger.info("Saved TF-IDF+LogReg bundle to %s", bundle_path)
        return bundle_path

    @classmethod
    def load(cls, bundle_path: str) -> "TfidfLogRegModel":
        bundle = joblib.load(bundle_path)
        model = cls()
        model.vectorizer = bundle["vectorizer"]
        model.classifier = bundle["classifier"]
        model._fitted = True
        return model
