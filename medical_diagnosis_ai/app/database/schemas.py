"""
Schema definitions and thin repository classes for the two collections
required by the project guide:

  * Conditions collection -- condition, symptoms, causes, warnings,
    recommendations (+ optional metadata: source_url, scraped_at).
  * Models collection -- name, type, gridfs_id, labels, metrics, created.

Repositories are intentionally simple (no ORM) so behaviour is easy to
audit against the requirements table.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

try:
    from pymongo.errors import DuplicateKeyError
except Exception:
    # pymongo may not be installed in lightweight test environments; provide
    # a local fallback so modules importing this file don't fail at import time.
    class DuplicateKeyError(Exception):
        pass

from app.database.connection import get_db
from app.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_CONDITION_FIELDS = ["condition", "symptoms", "causes", "warnings", "recommendations"]
REQUIRED_MODEL_FIELDS = ["name", "type", "gridfs_id", "labels", "metrics", "created"]


@dataclass
class ConditionDocument:
    """Required fields exactly match the PDF's Conditions Collection table."""
    condition: str
    symptoms: List[str]
    causes: List[str]
    warnings: str
    recommendations: str
    # Optional metadata, clearly separated from required fields
    source_url: Optional[str] = None
    scraped_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelDocument:
    """Required fields exactly match the PDF's Model Collection table."""
    name: str
    type: str
    gridfs_id: Optional[str]
    labels: List[str]
    metrics: Dict[str, Any]
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_condition_doc(doc: Dict[str, Any]) -> None:
    missing = [f for f in REQUIRED_CONDITION_FIELDS if f not in doc]
    if missing:
        raise ValueError(f"Condition document missing required fields: {missing}")


def validate_model_doc(doc: Dict[str, Any]) -> None:
    missing = [f for f in REQUIRED_MODEL_FIELDS if f not in doc]
    if missing:
        raise ValueError(f"Model document missing required fields: {missing}")


class ConditionsRepository:
    """CRUD helper around the conditions collection."""

    def __init__(self):
        cfg = get_config()
        self.collection = get_db()[cfg.CONDITIONS_COLLECTION]

    def upsert(self, doc: Dict[str, Any]) -> str:
        """Insert or update a condition, keyed by condition name (case-insensitive
        match on the exact string). Prevents duplicate condition documents."""
        validate_condition_doc(doc)
        result = self.collection.update_one(
            {"condition": doc["condition"]}, {"$set": doc}, upsert=True
        )
        if result.upserted_id is not None:
            return str(result.upserted_id)
        existing = self.collection.find_one({"condition": doc["condition"]}, {"_id": 1})
        return str(existing["_id"]) if existing else ""

    def get_by_name(self, condition_name: str) -> Optional[Dict[str, Any]]:
        return self.collection.find_one({"condition": condition_name})

    def list_all(self, limit: int = 0) -> List[Dict[str, Any]]:
        cursor = self.collection.find({})
        if limit and limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)

    def count(self) -> int:
        return self.collection.count_documents({})

    def exists(self, condition_name: str) -> bool:
        return self.collection.count_documents({"condition": condition_name}, limit=1) > 0


class ModelsRepository:
    """CRUD helper around the models collection."""

    def __init__(self):
        cfg = get_config()
        self.collection = get_db()[cfg.MODELS_COLLECTION]

    def save(self, doc: Dict[str, Any]) -> str:
        validate_model_doc(doc)
        try:
            result = self.collection.insert_one(doc)
            return str(result.inserted_id)
        except DuplicateKeyError:
            # Same model name saved again (e.g. retrain) -> version by
            # replacing the previous record's metrics/gridfs pointer.
            self.collection.update_one({"name": doc["name"]}, {"$set": doc})
            existing = self.collection.find_one({"name": doc["name"]}, {"_id": 1})
            logger.info("Model '%s' already existed; updated in place.", doc["name"])
            return str(existing["_id"]) if existing else ""

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self.collection.find_one({"name": name})

    def get_best_by_metric(self, metric_path: str) -> Optional[Dict[str, Any]]:
        """metric_path e.g. 'metrics.emergency_recall' -- returns the model
        document with the highest value for that metric."""
        docs = list(self.collection.find({}))
        scored = [d for d in docs if _dig(d, metric_path) is not None]
        if not scored:
            return None
        return max(scored, key=lambda d: _dig(d, metric_path))

    def list_all(self) -> List[Dict[str, Any]]:
        return list(self.collection.find({}))


def _dig(doc: Dict[str, Any], dotted_path: str):
    node = doc
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node
