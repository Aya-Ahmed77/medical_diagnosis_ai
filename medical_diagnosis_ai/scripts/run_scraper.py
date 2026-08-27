"""CLI: python scripts/run_scraper.py --limit 20 [--force-refresh]"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.scraping_service import run_scrape  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Scrape NHS Inform A-Z conditions into MongoDB.")
    parser.add_argument("--limit", type=int, default=None, help="Max new conditions to scrape (overrides .env).")
    parser.add_argument("--force-refresh", action="store_true", help="Re-scrape conditions already stored.")
    args = parser.parse_args()

    result = run_scrape(limit=args.limit, force_refresh=args.force_refresh)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
