"""
Flask application factory.

Registers the three required API blueprints (Data Handling, Preprocessing,
Model) and a basic health-check endpoint.
"""
from app.config import get_config
from app.database.connection import check_connection, ensure_indexes
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> "Flask":
    # Importing Flask and jsonify lazily so importing the app package for
    # unit tests that only exercise submodules (parser, preprocessing, etc.)
    # does not require Flask to be installed in the environment.
    from flask import Flask, jsonify

    cfg = get_config()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = cfg.SECRET_KEY

    from app.api.data_api import data_api
    from app.api.preprocessing_api import preprocessing_api
    from app.api.model_api import model_api

    app.register_blueprint(data_api)
    app.register_blueprint(preprocessing_api)
    app.register_blueprint(model_api)

    @app.route("/api/health", methods=["GET"])
    def health():
        mongo_ok = check_connection()
        return jsonify({
            "status": "ok" if mongo_ok else "degraded",
            "mongodb_connected": mongo_ok,
            "disclaimer": cfg.MEDICAL_DISCLAIMER,
        }), (200 if mongo_ok else 503)

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.exception("Unhandled server error")
        return jsonify({"error": "Internal server error"}), 500

    try:
        ensure_indexes()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure MongoDB indexes at startup (is MongoDB running?): %s", exc)

    return app
