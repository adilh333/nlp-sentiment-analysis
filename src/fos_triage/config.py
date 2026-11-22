"""Paths, source URLs and the sampling frame.

The Financial Ombudsman Service (FOS) publishes final decisions under FSMA.
This project reads that public search at a polite rate. It does not republish
the decisions database.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"
CACHE_HTML = CACHE / "html"
CACHE_PDF = CACHE / "pdf"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
AGGREGATES = DATA / "aggregates"
SAMPLE = DATA / "sample"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models" / "baseline"

SEARCH_URL = (
    "https://www.financial-ombudsman.org.uk"
    "/businesses/resolving-complaint/ombudsman-decisions/search"
)
PDF_URL = "https://www.financial-ombudsman.org.uk/decision/{reference}.pdf"
DECISIONS_HOME = (
    "https://www.financial-ombudsman.org.uk"
    "/businesses/resolving-complaint/ombudsman-decisions"
)
ROBOTS_URL = "https://www.financial-ombudsman.org.uk/robots.txt"

USER_AGENT = (
    "Mozilla/5.0 (compatible; fos-complaint-triage/0.1; "
    "research portfolio; rate-limited; "
    "+https://github.com/adilh333/nlp-sentiment-analysis)"
)

# Sector ids are the checkbox values on the public search form.
SECTORS: dict[int, str] = {
    1: "Banking, credit and mortgages",
    2: "Investment and pensions",
    3: "Insurance (excluding PPI)",
    4: "Payment protection insurance (PPI)",
    5: "Claims management",
    6: "Funeral plans",
}

# Population charts cover the modern register. The text corpus is the recent
# window a complaints team would actually tune a model on.
COUNT_YEARS = list(range(2018, 2027))
CORPUS_YEARS = list(range(2022, 2027))

# Per sector, per corpus year. Newest-first within the cell. Empty cells
# (for example PPI late in the window) simply stop early.
CORPUS_QUOTA_PER_CELL: dict[int, int] = {
    1: 280,  # banking, credit and mortgages
    2: 110,  # investment and pensions
    3: 150,  # insurance
    4: 70,  # PPI
    5: 50,  # claims management
    6: 30,  # funeral plans
}

# Keyword hit-counts for topical monitoring. These overlap; they are not a
# partition of the register.
THEMES: list[dict[str, str]] = [
    {
        "id": "motor_finance_commission",
        "keyword": "motor finance commission",
        "label": "Motor finance commission",
    },
    {
        "id": "discretionary_commission",
        "keyword": "discretionary commission",
        "label": "Discretionary commission",
    },
    {
        "id": "app_scam",
        "keyword": "authorised push payment",
        "label": "Authorised push payment scams",
    },
    {
        "id": "packaged_account",
        "keyword": "packaged bank account",
        "label": "Packaged bank accounts",
    },
    {
        "id": "section_75",
        "keyword": "section 75",
        "label": "Section 75 claims",
    },
    {
        "id": "irresponsible_lending",
        "keyword": "irresponsible lending",
        "label": "Irresponsible lending",
    },
]

MIN_INTAKE_CHARS = 160
RANDOM_STATE = 42
MIN_PRODUCT_COUNT = 40
