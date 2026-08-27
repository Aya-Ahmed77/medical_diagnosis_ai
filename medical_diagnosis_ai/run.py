"""
Application entry point.

Usage:
    python run.py

Reads API_HOST / API_PORT / FLASK_ENV from the environment (see
app/config.py and .env.example).
"""
from app import create_app
from app.config import get_config

app = create_app()

if __name__ == "__main__":
    cfg = get_config()
    debug = cfg.FLASK_ENV == "development"
    app.run(host=cfg.API_HOST, port=cfg.API_PORT, debug=debug)
