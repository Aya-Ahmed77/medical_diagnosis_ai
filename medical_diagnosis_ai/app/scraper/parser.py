"""
Pure parsing functions for NHS Inform pages.

Kept separate from nhs_scraper.py (which does the networking/orchestration)
so parsing logic can be unit tested against saved HTML fixtures without
any network access -- see tests/test_scraper.py.

NOTE ON SITE STRUCTURE: NHS Inform's markup can change over time. The
selectors below target the structure as of the guide's writing and use
several fallback strategies (heading-based section extraction) so the
parser degrades gracefully rather than crashing on minor markup drift.
"""
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Section headings we look for on a condition page, and which structured
# field each maps to. Matching is case-insensitive and substring-based
# because NHS Inform phrases headings slightly differently across pages
# (e.g. "Self-help guide", "Treatment", "How to look after yourself").
SECTION_HEADING_MAP = {
    "symptoms": ["symptom"],
    "causes": ["cause"],
    "warnings": [
        "when to get medical help", "emergency", "urgent advice",
        "immediate action required", "when to seek help", "warning",
    ],
    "recommendations": [
        "treatment", "self-help", "self help", "how to treat",
        "looking after yourself", "self-care", "management",
    ],
}


def extract_az_links(html: str, base_url: str) -> List[str]:
    """Parse the A-Z index page and return absolute URLs of every
    condition detail page linked from it.

    Strategy: the A-Z listing renders condition names as anchors inside
    the main content area. We take every anchor whose href looks like an
    illnesses-and-conditions detail page and is not itself an index/anchor
    link (e.g. '#a', '#b' letter jump links).
    """
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#"):
            continue
        if "illnesses-and-conditions" not in href:
            continue
        # Skip the A-Z index pages themselves and letter-jump anchors
        if href.rstrip("/").endswith("/a-to-z"):
            continue
        if re.search(r"/a-to-z/?(\?|$)", href):
            continue
        absolute = urljoin(base_url, href)
        links.add(absolute)

    logger.debug("Discovered %d candidate condition links", len(links))
    return sorted(links)


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip stray unicode artifacts from scraped text."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_section_text(soup: BeautifulSoup, heading_keywords: List[str]) -> str:
    """Find a heading (h2/h3) whose text contains one of heading_keywords,
    then concatenate the text of sibling content until the next heading."""
    headings = soup.find_all(["h2", "h3"])
    for heading in headings:
        heading_text = heading.get_text(" ", strip=True).lower()
        if any(kw in heading_text for kw in heading_keywords):
            collected = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ("h2", "h3"):
                    break
                text = sibling.get_text(" ", strip=True)
                if text:
                    collected.append(text)
            if collected:
                return _clean_text(" ".join(collected))
    return ""


def _extract_list_items(soup: BeautifulSoup, heading_keywords: List[str]) -> List[str]:
    """Like _extract_section_text but returns <li> items as a list, falling
    back to sentence-splitting the paragraph text if no <li> is present."""
    headings = soup.find_all(["h2", "h3"])
    for heading in headings:
        heading_text = heading.get_text(" ", strip=True).lower()
        if any(kw in heading_text for kw in heading_keywords):
            items: List[str] = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ("h2", "h3"):
                    break
                if sibling.name in ("ul", "ol"):
                    items.extend(
                        _clean_text(li.get_text(" ", strip=True))
                        for li in sibling.find_all("li")
                        if li.get_text(strip=True)
                    )
                elif sibling.name == "p":
                    text = sibling.get_text(" ", strip=True)
                    if text:
                        items.append(_clean_text(text))
            if items:
                return items
    return []


def parse_condition_page(html: str, url: str) -> Optional[Dict]:
    """Parse a single NHS Inform condition page into the structured schema
    required by the Conditions collection. Returns None if the page does
    not look like a valid condition page (e.g. it was a redirect to a
    listing page, or the essential content is missing).
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    if not title_tag:
        logger.warning("No <h1> found on %s -- skipping", url)
        return None
    condition_name = _clean_text(title_tag.get_text(" ", strip=True))
    if not condition_name:
        return None

    symptoms = _extract_list_items(soup, SECTION_HEADING_MAP["symptoms"])
    causes = _extract_list_items(soup, SECTION_HEADING_MAP["causes"])
    warnings = _extract_section_text(soup, SECTION_HEADING_MAP["warnings"])
    recommendations = _extract_section_text(soup, SECTION_HEADING_MAP["recommendations"])

    if not symptoms and not causes and not warnings and not recommendations:
        # Page matched the URL pattern but had none of the expected
        # sections -- likely not an actual condition detail page.
        logger.info("No structured sections found on %s -- skipping", url)
        return None

    return {
        "condition": condition_name,
        "symptoms": symptoms,
        "causes": causes,
        "warnings": warnings or "No specific emergency warning text was found on this page.",
        "recommendations": recommendations or "No specific self-care text was found on this page.",
        "source_url": url,
    }
