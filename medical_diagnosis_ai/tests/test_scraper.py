"""
Scraper parsing tests. These use inline HTML fixtures -- NOT live network
requests -- so they run offline and deterministically, per requirement 18
("do not require the full NHS website for basic unit tests").
"""
from app.scraper.parser import extract_az_links, parse_condition_page

AZ_INDEX_HTML = """
<html><body>
<nav>
  <a href="#a">A</a><a href="#b">B</a>
</nav>
<main>
  <a href="/illnesses-and-conditions/a-to-z">A-Z</a>
  <a href="/illnesses-and-conditions/asthma">Asthma</a>
  <a href="/illnesses-and-conditions/common-cold">Common cold</a>
</main>
</body></html>
"""

CONDITION_PAGE_HTML = """
<html><body>
<h1>Asthma</h1>
<h2>Symptoms of asthma</h2>
<ul>
  <li>Wheezing</li>
  <li>Shortness of breath</li>
  <li>Chest tightness</li>
</ul>
<h2>Causes of asthma</h2>
<ul>
  <li>Allergens</li>
  <li>Cold air</li>
</ul>
<h2>When to get emergency help</h2>
<p>Call 999 immediately if you have severe difficulty breathing.</p>
<h2>Treatment and self-help</h2>
<p>Use your reliever inhaler as prescribed and avoid known triggers.</p>
</body></html>
"""

EMPTY_PAGE_HTML = "<html><body><h1>Random Page</h1><p>Nothing structured here.</p></body></html>"


def test_extract_az_links_filters_index_and_anchors():
    links = extract_az_links(AZ_INDEX_HTML, "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/")
    assert any("asthma" in link for link in links)
    assert any("common-cold" in link for link in links)
    assert not any(link.rstrip("/").endswith("/a-to-z") for link in links)


def test_parse_condition_page_extracts_all_fields():
    parsed = parse_condition_page(CONDITION_PAGE_HTML, "https://www.nhsinform.scot/illnesses-and-conditions/asthma")
    assert parsed is not None
    assert parsed["condition"] == "Asthma"
    assert "Wheezing" in parsed["symptoms"]
    assert "Allergens" in parsed["causes"]
    assert "999" in parsed["warnings"]
    assert "inhaler" in parsed["recommendations"]
    assert parsed["source_url"].endswith("/asthma")


def test_parse_condition_page_returns_none_when_no_structured_sections():
    parsed = parse_condition_page(EMPTY_PAGE_HTML, "https://www.nhsinform.scot/illnesses-and-conditions/random")
    assert parsed is None
