"""Collect FOS decision metadata and intake text.

Two commands:

    python -m fos_triage.collect counts
    python -m fos_triage.collect corpus

`counts` records how many published decisions the search reports by year,
sector, outcome and topical keyword. Those totals are the population charts.
`corpus` downloads a stratified sample of decision PDFs plus the search
metadata. Intake text is built afterwards by `build_dataset`. Raw HTML and
PDFs stay in data/cache and are gitignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fos_triage.config import (
    AGGREGATES,
    CACHE,
    CORPUS_QUOTA_PER_CELL,
    CORPUS_YEARS,
    COUNT_YEARS,
    INTERIM,
    PDF_URL,
    SEARCH_URL,
    SECTORS,
    THEMES,
)
from fos_triage.http_client import PoliteClient
from fos_triage.parse_search import parse_result_count, parse_search_page

METADATA_PATH = INTERIM / "metadata.jsonl"
COUNTS_PATH = AGGREGATES / "population_counts.json"


def _year_bounds(year: int) -> tuple[str, str]:
    end = "2026-11-21" if year >= 2026 else f"{year}-12-31"
    return f"{year}-01-01", end


def _params(
    year: int | None = None,
    sector_id: int | None = None,
    upheld: bool | None = None,
    keyword: str | None = None,
    start: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[tuple[str, str]]:
    if year is not None and date_from is None:
        date_from, date_to = _year_bounds(year)
    params: list[tuple[str, str]] = [("Sort", "date")]
    if date_from:
        params.append(("DateFrom", date_from))
    if date_to:
        params.append(("DateTo", date_to))
    if sector_id is not None:
        params.append((f"IndustrySectorID[{sector_id}]", str(sector_id)))
    if upheld is True:
        params.append(("IsUpheld[1]", "1"))
    elif upheld is False:
        params.append(("IsUpheld[0]", "0"))
    if keyword:
        params.append(("Keyword", keyword))
    if start is not None:
        params.append(("Start", str(start)))
    return params


def _cache_key(prefix: str, params: list[tuple[str, str]]) -> str:
    joined = "&".join(f"{key}={value}" for key, value in params)
    return f"{prefix}:{joined}"


def fetch_search(client: PoliteClient, params: list[tuple[str, str]]) -> str:
    raw = client.get(SEARCH_URL, params=params, cache_key=_cache_key("search", params))
    return raw.decode("utf-8", "replace")


def result_count(client: PoliteClient, params: list[tuple[str, str]]) -> int | None:
    html = fetch_search(client, params)
    return parse_result_count(html)


def collect_counts(client: PoliteClient) -> dict:
    by_year_sector = []
    for year in COUNT_YEARS:
        for sector_id, sector in SECTORS.items():
            total = result_count(client, _params(year=year, sector_id=sector_id))
            upheld = result_count(client, _params(year=year, sector_id=sector_id, upheld=True))
            not_upheld = result_count(client, _params(year=year, sector_id=sector_id, upheld=False))
            by_year_sector.append(
                {
                    "year": year,
                    "sector_id": sector_id,
                    "sector": sector,
                    "total": total,
                    "upheld": upheld,
                    "not_upheld": not_upheld,
                }
            )
            print(f"counts {year} {sector}: total={total} upheld={upheld}", flush=True)
        overall = result_count(client, _params(year=year))
        upheld = result_count(client, _params(year=year, upheld=True))
        not_upheld = result_count(client, _params(year=year, upheld=False))
        by_year_sector.append(
            {
                "year": year,
                "sector_id": 0,
                "sector": "All sectors",
                "total": overall,
                "upheld": upheld,
                "not_upheld": not_upheld,
            }
        )
        print(f"counts {year} ALL: total={overall} upheld={upheld}", flush=True)

    by_year_theme = []
    for year in COUNT_YEARS:
        for theme in THEMES:
            total = result_count(client, _params(year=year, keyword=theme["keyword"]))
            by_year_theme.append(
                {
                    "year": year,
                    "theme": theme["id"],
                    "label": theme["label"],
                    "keyword": theme["keyword"],
                    "total": total,
                }
            )
            print(f"theme {year} {theme['id']}: {total}", flush=True)

    payload = {
        "source": SEARCH_URL,
        "note": (
            "Figures are the hit counts reported by the FOS decisions search. "
            "They count published final ombudsman decisions, not every complaint "
            "a firm received. Keyword counts overlap and are not a partition."
        ),
        "date_to_cap": "2026-11-21",
        "by_year_sector": by_year_sector,
        "by_year_theme": by_year_theme,
    }
    AGGREGATES.mkdir(parents=True, exist_ok=True)
    COUNTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_seen(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["reference"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def _cell_counts(path: Path) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                year = int(str(row.get("date") or "")[:4])
                counts[(int(row["sector_id"]), year)] = counts.get((int(row["sector_id"]), year), 0) + 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return counts


def _pdf_url(reference: str, href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("decision/"):
        return f"https://www.financial-ombudsman.org.uk/{href}"
    return PDF_URL.format(reference=reference)


def collect_corpus(client: PoliteClient, per_cell: dict[int, int] | None = None, years: list[int] | None = None) -> int:
    """Download search metadata and PDFs. Text extraction is a separate step."""
    quotas = per_cell or CORPUS_QUOTA_PER_CELL
    years = years or CORPUS_YEARS
    INTERIM.mkdir(parents=True, exist_ok=True)
    seen = _load_seen(METADATA_PATH)
    already = _cell_counts(METADATA_PATH)
    written = 0
    with METADATA_PATH.open("a", encoding="utf-8") as handle:
        for sector_id, quota in quotas.items():
            sector = SECTORS[sector_id]
            for year in years:
                got = already.get((sector_id, year), 0)
                if got >= quota:
                    print(f"cell {year} {sector}: already {got}, skipping", flush=True)
                    continue
                start = 0
                empty_pages = 0
                while got < quota and empty_pages < 2 and start <= 8000:
                    params = _params(year=year, sector_id=sector_id, start=start)
                    html = fetch_search(client, params)
                    rows = parse_search_page(html)
                    if not rows:
                        break
                    fresh = 0
                    for row in rows:
                        reference = row["reference"]
                        if reference in seen:
                            continue
                        fresh += 1
                        seen.add(reference)
                        url = _pdf_url(reference, row.get("pdf_href") or "")
                        pdf_ok = True
                        try:
                            client.get(url, cache_key=f"pdf:{reference}")
                        except Exception as exc:
                            pdf_ok = False
                            print(f"pdf failed {reference}: {exc}", flush=True)
                        record = {
                            "reference": reference,
                            "date": row.get("date"),
                            "business": row.get("business"),
                            "outcome": row.get("outcome"),
                            "upheld": row.get("upheld"),
                            "sector_id": sector_id,
                            "sector": sector,
                            "product": row.get("product"),
                            "snippet": row.get("snippet") or "",
                            "pdf_url": url,
                            "pdf_ok": pdf_ok,
                            "source_register": SEARCH_URL,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        got += 1
                        written += 1
                        if written % 50 == 0:
                            print(f"downloaded {written} last={reference} {record['date']} {record['product']}", flush=True)
                        if got >= quota:
                            break
                    if fresh == 0:
                        empty_pages += 1
                    start += 10
                print(f"cell {year} {sector}: downloaded {got}", flush=True)
    print(f"metadata rows this run: {written} file={METADATA_PATH}", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect FOS decisions")
    sub = parser.add_subparsers(dest="command", required=True)
    counts = sub.add_parser("counts")
    counts.add_argument("--delay", type=float, default=0.75)
    corpus = sub.add_parser("corpus")
    corpus.add_argument("--delay", type=float, default=0.75)
    corpus.add_argument("--max-per-cell", type=int, default=None, help="Override every sector quota, for a smoke run")
    args = parser.parse_args()
    client = PoliteClient(delay_s=args.delay, cache_dir=CACHE)
    client.check_robots()
    if args.command == "counts":
        collect_counts(client)
    else:
        quotas = None
        if args.max_per_cell is not None:
            quotas = {sector_id: args.max_per_cell for sector_id in SECTORS}
        collect_corpus(client, per_cell=quotas)
    print(f"http requests={client.requests_made} cache_hits={client.cache_hits}", flush=True)


if __name__ == "__main__":
    main()
