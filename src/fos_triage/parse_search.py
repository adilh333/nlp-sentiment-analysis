"""Parse one FOS decisions search-results page."""

from __future__ import annotations

import re
from datetime import datetime

from bs4 import BeautifulSoup

_COUNT_RE = re.compile(r"returned\s+([0-9,]+)\s+results?", re.I)
_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})")
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_result_count(html: str) -> int | None:
    match = _COUNT_RE.search(html)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def parse_display_date(value: str) -> str | None:
    match = _DATE_RE.search(value.strip())
    if not match:
        return None
    month = _MONTHS.get(match.group(2)[:3].lower())
    if not month:
        return None
    try:
        return datetime(int(match.group(3)), month, int(match.group(1))).date().isoformat()
    except ValueError:
        return None


def parse_outcome(value: str) -> tuple[str, int | None]:
    text = " ".join(value.lower().split())
    if "not upheld" in text:
        return "not_upheld", 0
    if "upheld" in text:
        return "upheld", 1
    return "unknown", None


def _clean_snippet(node) -> str:
    for tag in node.find_all(["strong", "br"]):
        tag.decompose()
    return " ".join(node.get_text(" ", strip=True).split())


def parse_search_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    for anchor in soup.select("a.search-result"):
        href = anchor.get("href") or ""
        heading = anchor.find("h3")
        reference = ""
        if heading:
            reference = heading.get_text(" ", strip=True).replace("Decision Reference", "").strip()
        if not reference and "/decision/" in href:
            reference = href.rsplit("/", 1)[-1].replace(".pdf", "")
        info = anchor.select_one(".search-result__info-main")
        date_iso = None
        business = None
        outcome_label, upheld = "unknown", None
        if info:
            emphasised = info.find("em")
            if emphasised:
                date_iso = parse_display_date(emphasised.get_text(" ", strip=True))
            # The info line is: date, business, outcome, separated by spans.
            text = " ".join(info.get_text("\n", strip=True).split("\n"))
            parts = [part.strip() for part in info.get_text("\n", strip=True).split("\n") if part.strip()]
            # parts typically: [date, business, outcome] or [date, outcome]
            if len(parts) >= 3:
                business = parts[1]
                outcome_label, upheld = parse_outcome(parts[2])
            elif len(parts) == 2:
                outcome_label, upheld = parse_outcome(parts[1])
            else:
                outcome_label, upheld = parse_outcome(text)
        tag = anchor.select_one(".search-result__tag")
        product = tag.get_text(" ", strip=True) if tag else None
        desc = anchor.select_one(".search-result__desc")
        snippet = _clean_snippet(desc) if desc else ""
        if not reference:
            continue
        rows.append(
            {
                "reference": reference,
                "pdf_href": href,
                "date": date_iso,
                "business": business,
                "outcome": outcome_label,
                "upheld": upheld,
                "product": product,
                "snippet": snippet,
            }
        )
    return rows
