"""
Central configuration module.

Every configurable value (Mongo URI, scraping limits, model names,
training hyperparameters, API port, etc.) is read from environment
variables so nothing is hard-coded. See .env.example for the full list
of variables this project understands.

Usage:
    from app.config import get_config
    cfg = get_config()
    cfg.MONGO_URI
"""
import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads a local .env file if present; no-op otherwise
except ImportError:
    # python-dotenv is a convenience only -- the app must still work
    # from real environment variables if it isn't installed.
    pass


def _bool(env_name: str, default: bool) -> bool:
    val = os.environ.get(env_name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int(env_name: str, default: int) -> int:
    val = os.environ.get(env_name)
    return int(val) if val is not None and val != "" else default


def _float(env_name: str, default: float) -> float:
    val = os.environ.get(env_name)
    return float(val) if val is not None and val != "" else default


def _list(env_name: str, default: List[str]) -> List[str]:
    val = os.environ.get(env_name)
    if not val:
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


@dataclass
class Config:
    # --- Flask ---
    FLASK_ENV: str = os.environ.get("FLASK_ENV", "development")
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = field(default_factory=lambda: _int("API_PORT", 5000))
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-change-me")

    # --- MongoDB ---
    MONGO_URI: str = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.environ.get("MONGO_DB_NAME", "medical_diagnosis_ai")
    CONDITIONS_COLLECTION: str = os.environ.get("CONDITIONS_COLLECTION", "conditions")
    MODELS_COLLECTION: str = os.environ.get("MODELS_COLLECTION", "models")

    # --- Scraper ---
    NHS_BASE_URL: str = os.environ.get(
        "NHS_BASE_URL", "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/"
    )
    SCRAPER_USER_AGENT: str = os.environ.get(
        "SCRAPER_USER_AGENT",
        "MedicalDiagnosisAI-EducationalProject/1.0 (+contact: student-project)",
    )
    SCRAPER_REQUEST_TIMEOUT: int = field(default_factory=lambda: _int("SCRAPER_REQUEST_TIMEOUT", 10))
    SCRAPER_MAX_RETRIES: int = field(default_factory=lambda: _int("SCRAPER_MAX_RETRIES", 3))
    SCRAPER_RATE_LIMIT_SECONDS: float = field(default_factory=lambda: _float("SCRAPER_RATE_LIMIT_SECONDS", 1.0))
    SCRAPER_MAX_CONDITIONS: int = field(default_factory=lambda: _int("SCRAPER_MAX_CONDITIONS", 20))
    # SCRAPER_MAX_CONDITIONS <= 0 means "no limit, scrape everything found"

    # --- Preprocessing / dataset ---
    MIN_SYMPTOM_TOKENS: int = field(default_factory=lambda: _int("MIN_SYMPTOM_TOKENS", 2))
    RANDOM_SEED: int = field(default_factory=lambda: _int("RANDOM_SEED", 42))
    TRAIN_SPLIT: float = field(default_factory=lambda: _float("TRAIN_SPLIT", 0.7))
    VAL_SPLIT: float = field(default_factory=lambda: _float("VAL_SPLIT", 0.15))
    TEST_SPLIT: float = field(default_factory=lambda: _float("TEST_SPLIT", 0.15))
    SYNTHETIC_EXAMPLES_PER_CONDITION: int = field(
        default_factory=lambda: _int("SYNTHETIC_EXAMPLES_PER_CONDITION", 8)
    )

    # --- Baseline model (TF-IDF + Logistic Regression) ---
    TFIDF_MAX_FEATURES: int = field(default_factory=lambda: _int("TFIDF_MAX_FEATURES", 5000))
    TFIDF_NGRAM_MAX: int = field(default_factory=lambda: _int("TFIDF_NGRAM_MAX", 2))
    LOGREG_MAX_ITER: int = field(default_factory=lambda: _int("LOGREG_MAX_ITER", 1000))

    # --- RNN / LSTM ---
    VOCAB_SIZE: int = field(default_factory=lambda: _int("VOCAB_SIZE", 8000))
    MAX_SEQUENCE_LENGTH: int = field(default_factory=lambda: _int("MAX_SEQUENCE_LENGTH", 64))
    EMBEDDING_DIM: int = field(default_factory=lambda: _int("EMBEDDING_DIM", 128))
    RNN_UNITS: int = field(default_factory=lambda: _int("RNN_UNITS", 64))
    LSTM_UNITS: int = field(default_factory=lambda: _int("LSTM_UNITS", 64))
    DL_EPOCHS: int = field(default_factory=lambda: _int("DL_EPOCHS", 20))
    DL_BATCH_SIZE: int = field(default_factory=lambda: _int("DL_BATCH_SIZE", 16))
    DL_LEARNING_RATE: float = field(default_factory=lambda: _float("DL_LEARNING_RATE", 1e-3))
    EARLY_STOPPING_PATIENCE: int = field(default_factory=lambda: _int("EARLY_STOPPING_PATIENCE", 3))

    # --- Transformer ---
    TRANSFORMER_MODEL_NAME: str = os.environ.get(
        "TRANSFORMER_MODEL_NAME", "dmis-lab/biobert-base-cased-v1.2"
    )
    TRANSFORMER_FALLBACK_MODEL_NAME: str = os.environ.get(
        "TRANSFORMER_FALLBACK_MODEL_NAME", "bert-base-uncased"
    )
    TRANSFORMER_MAX_LENGTH: int = field(default_factory=lambda: _int("TRANSFORMER_MAX_LENGTH", 128))
    TRANSFORMER_EPOCHS: int = field(default_factory=lambda: _int("TRANSFORMER_EPOCHS", 4))
    TRANSFORMER_BATCH_SIZE: int = field(default_factory=lambda: _int("TRANSFORMER_BATCH_SIZE", 8))
    TRANSFORMER_LEARNING_RATE: float = field(default_factory=lambda: _float("TRANSFORMER_LEARNING_RATE", 2e-5))

    # --- Evaluation ---
    TOP_K_VALUES: List[int] = field(default_factory=lambda: [1, 3, 5])
    EMERGENCY_KEYWORDS: List[str] = field(
        default_factory=lambda: _list(
            "EMERGENCY_KEYWORDS",
            [
                "999", "a&e", "accident and emergency", "emergency department",
                "immediately", "urgent", "life-threatening", "call 999",
                "seek immediate", "call an ambulance", "chest pain",
                "difficulty breathing", "severe bleeding",
            ],
        )
    )

    # --- Prediction ---
    PREDICTION_TOP_N: int = field(default_factory=lambda: _int("PREDICTION_TOP_N", 5))
    MEDICAL_DISCLAIMER: str = (
        "This system provides educational, model-generated possibilities based on "
        "the symptoms you entered. It is NOT a medical diagnosis and must NOT be "
        "used as a substitute for professional medical advice. If you believe you "
        "are having a medical emergency, contact emergency services immediately."
    )

    # --- Artifact / storage paths (local filesystem, used before/around GridFS) ---
    DATA_DIR: str = os.environ.get("DATA_DIR", "data")
    RAW_DATA_DIR: str = os.environ.get("RAW_DATA_DIR", "data/raw")
    PROCESSED_DATA_DIR: str = os.environ.get("PROCESSED_DATA_DIR", "data/processed")
    ARTIFACTS_DIR: str = os.environ.get("ARTIFACTS_DIR", "data/artifacts")


_config_instance = None


def get_config() -> Config:
    """Return a process-wide singleton Config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
