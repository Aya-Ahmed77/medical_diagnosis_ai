"""Service layer wrapping the scraper for use by the Data Handling API."""
from typing import Optional

from app.scraper.nhs_scraper import NHSScraper, ScrapeResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_scrape(limit: Optional[int] = None, force_refresh: bool = False) -> dict:
    scraper = NHSScraper()
    result: ScrapeResult = scraper.run(limit=limit, force_refresh=force_refresh)
    return {
        "total_links_found": result.total_links_found,
        "scraped": result.scraped,
        "skipped_existing": result.skipped_existing,
        "failed": result.failed,
        "failed_urls": result.failed_urls[:20],  # cap response size
    }
