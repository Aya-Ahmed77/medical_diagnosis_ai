"""
MongoDB connection management.

Provides a single shared PyMongo client/database for the whole app,
plus a thin helper to create indexes. Nothing here hard-codes a URI --
everything comes from app.config.
"""
from functools import lru_cache

try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.database import Database
    from pymongo.errors import PyMongoError
except Exception:  # pymongo may be unavailable in minimal test environments
    MongoClient = None
    ASCENDING = None
    Database = object
    PyMongoError = Exception

from app.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_client():
    """Return a process-wide singleton MongoClient.

    If pymongo is not installed, this will raise when an attempt is made to
    actually use the client. Tests that mock DB access can still import
    this module without having pymongo installed.
    """
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed in this environment")
    cfg = get_config()
    client = MongoClient(cfg.MONGO_URI, serverSelectionTimeoutMS=5000)
    return client


def get_db():
    """Return the configured application database."""
    return get_client()[get_config().MONGO_DB_NAME]


def check_connection() -> bool:
    """Ping MongoDB, returning True/False rather than raising, so callers
    (e.g. API health checks) can degrade gracefully."""
    try:
        get_client().admin.command("ping")
        return True
    except Exception as exc:
        logger.error("MongoDB connection check failed: %s", exc)
        return False


def ensure_indexes() -> None:
    """Create indexes required by the schema (idempotent -- safe to call
    on every startup). If pymongo is unavailable, this is a no-op."""
    if MongoClient is None:
        logger.warning("pymongo not installed; skipping ensure_indexes()")
        return

    cfg = get_config()
    db = get_db()

    conditions = db[cfg.CONDITIONS_COLLECTION]
    conditions.create_index([("condition", ASCENDING)], unique=True, name="uniq_condition")
    conditions.create_index([("source_url", ASCENDING)], name="idx_source_url")

    models = db[cfg.MODELS_COLLECTION]
    models.create_index([("name", ASCENDING)], unique=True, name="uniq_model_name")
    models.create_index([("type", ASCENDING)], name="idx_model_type")
    models.create_index([("created", ASCENDING)], name="idx_model_created")

    logger.info("MongoDB indexes ensured on '%s' and '%s'.", cfg.CONDITIONS_COLLECTION, cfg.MODELS_COLLECTION)
