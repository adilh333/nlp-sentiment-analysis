"""Turn cached decision PDFs into an intake-text modelling table.

The full table is written to data/processed and is gitignored. A short sample
and a metadata frame with no complaint prose are written for the repository.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from fos_triage.config import AGGREGATES, CACHE, INTERIM, MIN_INTAKE_CHARS, PROCESSED, REPORTS, SAMPLE
from fos_triage.extract import intake_from_text
from fos_triage.firm_type import firm_type
from fos_triage.themes import tag_themes

METADATA_PATH = INTERIM / "metadata.jsonl"
PDF_DIR = CACHE / "pdf"

# The register uses more than one display name for the same product family.
# PPI stays separate from other insurance. Pensions stay separate from investments.
PRODUCT_NORMALISE = {
    "Insurance (excluding PPI)": "Insurance",
    "Payment protection insurance (PPI)": "PPI",
    "Funeral Planning Services": "Funeral plans",
}


def pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return ""
    try:
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def load_metadata(path: Path = METADATA_PATH) -> list[dict]:
    rows = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            reference = row.get("reference")
            if not reference or reference in seen:
                continue
            seen.add(reference)
            rows.append(row)
    return rows


def build_records(metadata: list[dict]) -> tuple[list[dict], dict]:
    records: list[dict] = []
    audit = {
        "metadata_rows": len(metadata),
        "missing_pdf": 0,
        "empty_pdf_text": 0,
        "used_snippet": 0,
        "dropped_short": 0,
        "dropped_leak": 0,
        "dropped_label": 0,
        "kept": 0,
    }
    for row in metadata:
        reference = row["reference"]
        path = PDF_DIR / f"{reference}.pdf"
        raw = pdf_text(path) if path.exists() else ""
        if not path.exists():
            audit["missing_pdf"] += 1
        elif not raw.strip():
            audit["empty_pdf_text"] += 1
        text_source = "pdf"
        intake = intake_from_text(raw) if raw.strip() else None
        if intake is None or intake.n_chars < MIN_INTAKE_CHARS or intake.leak_flags:
            snippet_intake = intake_from_text(row.get("snippet") or "")
            # Prefer a clean snippet over a leaky or tiny PDF extract.
            pdf_unusable = (
                intake is None
                or intake.n_chars < MIN_INTAKE_CHARS
                or bool(intake.leak_flags)
            )
            if pdf_unusable and snippet_intake.n_chars >= MIN_INTAKE_CHARS and not snippet_intake.leak_flags:
                intake = snippet_intake
                text_source = "snippet"
                audit["used_snippet"] += 1
        if intake is None or intake.n_chars < MIN_INTAKE_CHARS:
            audit["dropped_short"] += 1
            continue
        if intake.leak_flags:
            audit["dropped_leak"] += 1
            continue
        if row.get("upheld") is None or not row.get("product") or not row.get("date"):
            audit["dropped_label"] += 1
            continue
        business = row.get("business")
        product_raw = row.get("product")
        product = PRODUCT_NORMALISE.get(product_raw, product_raw)
        # On PPI-sector results the product chip often still says "Insurance".
        # The sector filter is the register's own PPI flag, and the complaint
        # text is about PPI. Keep PPI as its own code.
        if row.get("sector") == "Payment protection insurance (PPI)":
            product = "PPI"
        record = {
            "reference": reference,
            "date": row.get("date"),
            "business": business,
            "firm_type": firm_type(business),
            "outcome": row.get("outcome"),
            "upheld": int(row["upheld"]),
            "sector_id": row.get("sector_id"),
            "sector": row.get("sector"),
            "product_raw": product_raw,
            "product": product,
            "model_text": intake.model_text,
            "complaint_text": intake.complaint,
            "background_text": intake.background,
            "n_chars": intake.n_chars,
            "sentences_removed": intake.sentences_removed,
            "text_source": text_source,
            "themes": tag_themes(intake.model_text),
            "pdf_url": row.get("pdf_url"),
        }
        records.append(record)
        audit["kept"] += 1
    return records, audit


def write_outputs(records: list[dict], audit: dict) -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    SAMPLE.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise SystemExit("No usable decisions were extracted")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "reference"]).reset_index(drop=True)
    frame.to_parquet(PROCESSED / "decisions.parquet", index=False)

    public = frame.drop(columns=["model_text", "complaint_text", "background_text"]).copy()
    public["date"] = public["date"].dt.date.astype(str)
    public["themes"] = public["themes"].apply(lambda values: "|".join(values))
    public.to_csv(PROCESSED / "analysis_frame.csv", index=False)
    AGGREGATES.mkdir(parents=True, exist_ok=True)
    public.to_csv(AGGREGATES / "case_index.csv", index=False)

    # Small quoted sample so a checkout can exercise the code. Complaint
    # sections only, capped, with the source PDF link.
    parts = []
    for _, group in frame.groupby("product"):
        upheld = group[group["upheld"] == 1].head(2)
        not_upheld = group[group["upheld"] == 0].head(2)
        parts.append(pd.concat([upheld, not_upheld]))
    sample = pd.concat(parts).drop_duplicates("reference").sort_values("date")
    sample_rows = []
    for row in sample.to_dict(orient="records"):
        complaint = " ".join(str(row["complaint_text"]).split())
        sample_rows.append(
            {
                "reference": row["reference"],
                "date": row["date"].date().isoformat(),
                "business": row["business"],
                "firm_type": row["firm_type"],
                "outcome": row["outcome"],
                "upheld": int(row["upheld"]),
                "sector": row["sector"],
                "product": row["product"],
                "model_text": complaint[:700],
                "pdf_url": row["pdf_url"],
                "note": "Quotation of the complaint section only, for code demonstration. Read the decision on the FOS site.",
            }
        )
    sample_path = SAMPLE / "decisions.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit["sample_rows"] = len(sample_rows)
    audit["date_min"] = frame["date"].min().date().isoformat()
    audit["date_max"] = frame["date"].max().date().isoformat()
    audit["upheld_rate"] = float(frame["upheld"].mean())
    audit["median_chars"] = float(frame["n_chars"].median())
    audit["products"] = {str(k): int(v) for k, v in frame["product"].value_counts().items()}
    (PROCESSED / "extraction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "extraction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return frame


def main() -> None:
    metadata = load_metadata()
    records, audit = build_records(metadata)
    write_outputs(records, audit)


if __name__ == "__main__":
    main()
