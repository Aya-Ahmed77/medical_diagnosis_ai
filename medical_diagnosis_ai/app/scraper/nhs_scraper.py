"""
NHS Inform A-Z scraper.

Orchestrates: fetch A-Z index -> discover condition links -> visit each
condition page -> parse -> store in MongoDB (via ConditionsRepository).

Robustness features (per project requirements):
  * requests.Session with a real User-Agent
  * configurable timeout
  * retry with backoff on transient failures
  * configurable delay between requests (politeness / rate limiting)
  * duplicate prevention (skips conditions already in MongoDB unless
    force_refresh=True)
  * graceful per-page failure handling (one bad page doesn't kill the run)
  * configurable maximum number of conditions, for dev/testing

IMPORTANT: this module makes real HTTP requests to nhsinform.scot. In
network-restricted environments (e.g. this build sandbox) those requests
will fail -- run() will raise/report that clearly rather than silently
fabricating data. See README "Known limitations".
"""
import time
from dataclasses import dataclass
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from app.config import get_config
from app.database.schemas import ConditionsRepository
from app.scraper.parser import extract_az_links, parse_condition_page
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScrapeResult:
    total_links_found: int
    scraped: int
    skipped_existing: int
    failed: int
    failed_urls: List[str]


class NHSScraper:
    def __init__(self, repository: Optional[ConditionsRepository] = None):
        self.cfg = get_config()
        self.repository = repository or ConditionsRepository()
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": self.cfg.SCRAPER_USER_AGENT})
        retries = Retry(
            total=self.cfg.SCRAPER_MAX_RETRIES,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.cfg.SCRAPER_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            logger.error("Request failed for %s: %s", url, exc)
            return None

    def discover_condition_links(self) -> List[str]:
        html = self._get(self.cfg.NHS_BASE_URL)
        if html is None:
            raise ConnectionError(
                f"Could not reach NHS Inform A-Z index at {self.cfg.NHS_BASE_URL}. "
                "Check network access / the configured NHS_BASE_URL."
            )
        return extract_az_links(html, self.cfg.NHS_BASE_URL)

    def run(self, limit: Optional[int] = None, force_refresh: bool = False) -> ScrapeResult:
        """Scrape condition pages and store them in MongoDB.

        Args:
            limit: max number of *new* conditions to scrape this run.
                Defaults to cfg.SCRAPER_MAX_CONDITIONS (<=0 means no limit).
            force_refresh: if True, re-scrape and overwrite conditions
                that already exist in MongoDB instead of skipping them.
        """
        effective_limit = limit if limit is not None else self.cfg.SCRAPER_MAX_CONDITIONS

        links = self.discover_condition_links()
        logger.info("Discovered %d condition links from A-Z index.", len(links))

        scraped = 0
        skipped = 0
        failed = 0
        failed_urls: List[str] = []

        for url in links:
            if effective_limit and effective_limit > 0 and scraped >= effective_limit:
                logger.info("Reached configured limit of %d conditions -- stopping.", effective_limit)
                break

            html = self._get(url)
            if html is None:
                failed += 1
                failed_urls.append(url)
                continue

            parsed = parse_condition_page(html, url)
            if parsed is None:
                failed += 1
                failed_urls.append(url)
                continue

            if not force_refresh and self.repository.exists(parsed["condition"]):
                logger.debug("Skipping already-stored condition: %s", parsed["condition"])
                skipped += 1
                continue

            self.repository.upsert(parsed)
            scraped += 1
            logger.info("Stored condition: %s", parsed["condition"])

            time.sleep(self.cfg.SCRAPER_RATE_LIMIT_SECONDS)

        return ScrapeResult(
            total_links_found=len(links),
            scraped=scraped,
            skipped_existing=skipped,
            failed=failed,
            failed_urls=failed_urls,
        )
