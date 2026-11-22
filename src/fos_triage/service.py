"""Score one complaint the way the demo and the API do."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib

from fos_triage.config import MODELS
from fos_triage.extract import intake_from_text
from fos_triage.firm_type import FIRM_TYPE_LABELS, firm_type
from fos_triage.summarize import extractive_summary, key_issues

_DECISION_MARK = re.compile(r"(?i)what i(?:'ve| have) decided|my final decision|putting things right")

DISCLAIMER = (
    "This is a triage aid trained on published Financial Ombudsman Service decisions. "
    "It is not a final response under DISP, not a fair-and-reasonable determination, "
    "and not a reason on its own to accept or reject a complaint. A person has to read the case."
)


class TriageService:
    def __init__(self, product_model, uphold_model, metadata: dict):
        self.product_model = product_model
        self.uphold_model = uphold_model
        self.metadata = metadata

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "TriageService":
        directory = Path(model_dir) if model_dir else MODELS
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return cls(
            product_model=joblib.load(directory / "product.joblib"),
            uphold_model=joblib.load(directory / "uphold.joblib"),
            metadata=metadata,
        )

    def prepare_text(self, text: str) -> tuple[str, bool]:
        if _DECISION_MARK.search(text or ""):
            intake = intake_from_text(text)
            if len(intake.model_text) >= 80:
                return intake.model_text, True
        return " ".join((text or "").split()), False

    def triage(self, text: str, business_name: str | None = None) -> dict:
        model_text, stripped = self.prepare_text(text)
        product_proba = self.product_model.predict_proba([model_text])[0]
        classes = [str(label) for label in self.product_model.named_steps["clf"].classes_]
        order = product_proba.argsort()[::-1]
        top3 = [
            {"label": classes[index], "confidence": round(float(product_proba[index]), 3)}
            for index in order[:3]
        ]
        uphold_p = float(self.uphold_model.predict_proba([model_text])[0])
        threshold = float(self.metadata.get("uphold_threshold", getattr(self.uphold_model, "threshold", 0.5)))
        high_cut = float(self.metadata.get("priority_cuts", {}).get("high", 0.7))
        if uphold_p >= high_cut:
            priority = "high"
        elif uphold_p >= threshold:
            priority = "medium"
        else:
            priority = "standard"
        firm = firm_type(business_name) if business_name else None
        return {
            "product": {
                "label": top3[0]["label"],
                "confidence": top3[0]["confidence"],
                "top3": top3,
            },
            "uphold": {
                "probability": round(uphold_p, 3),
                "priority": priority,
                "threshold": round(threshold, 3),
            },
            "summary": extractive_summary(model_text),
            "key_issues": key_issues(model_text),
            "firm_type": FIRM_TYPE_LABELS.get(firm, firm) if firm else None,
            "intake_text_used": stripped,
            "disclaimer": self.metadata.get("disclaimer", DISCLAIMER),
        }
