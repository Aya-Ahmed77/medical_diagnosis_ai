"""
Data Handling API (project requirement 14).

Endpoints:
  POST /api/data/scrape        -- run the NHS scraper, insert/update MongoDB
  GET  /api/data/conditions     -- list stored conditions (paginated via `limit`)
  GET  /api/data/conditions/<name> -- fetch a single condition document
"""
from flask import Blueprint, jsonify, request

from app.database.schemas import ConditionsRepository
from app.services.scraping_service import run_scrape
from app.utils.logger import get_logger
from app.utils.validators import ValidationError, optional_int

logger = get_logger(__name__)
data_api = Blueprint("data_api", __name__, url_prefix="/api/data")


@data_api.errorhandler(ValidationError)
def _handle_validation_error(err):
    return jsonify({"error": str(err)}), 400


@data_api.route("/scrape", methods=["POST"])
def scrape():
    """Start the NHS scraper. Body (all optional):
        { "limit": 20, "force_refresh": false }
    """
    payload = request.get_json(silent=True) or {}
    ok, limit = optional_int(payload, "limit")
    if not ok:
        raise ValidationError("'limit' must be an integer.")
    force_refresh = bool(payload.get("force_refresh", False))

    try:
        result = run_scrape(limit=limit, force_refresh=force_refresh)
        return jsonify({"status": "completed", **result}), 200
    except ConnectionError as exc:
        logger.error("Scrape failed: %s", exc)
        return jsonify({"status": "failed", "error": str(exc)}), 502
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected scraping error")
        return jsonify({"status": "failed", "error": str(exc)}), 500


@data_api.route("/conditions", methods=["GET"])
def list_conditions():
    limit = request.args.get("limit", default=0, type=int)
    repo = ConditionsRepository()
    conditions = repo.list_all(limit=limit)
    for c in conditions:
        c["_id"] = str(c["_id"])
    return jsonify({"count": len(conditions), "conditions": conditions}), 200


@data_api.route("/conditions/<string:name>", methods=["GET"])
def get_condition(name: str):
    repo = ConditionsRepository()
    doc = repo.get_by_name(name)
    if not doc:
        return jsonify({"error": f"Condition '{name}' not found."}), 404
    doc["_id"] = str(doc["_id"])
    return jsonify(doc), 200
