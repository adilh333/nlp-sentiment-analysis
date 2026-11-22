from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fos_triage.api import app, get_service
from fos_triage.modeling import PlattCalibrator, UpholdBundle, time_split
from fos_triage.service import TriageService


def test_time_split_does_not_share_dates():
    start = datetime(2022, 1, 1)
    rows = []
    for offset in range(100):
        rows.append(
            {
                "date": start + timedelta(days=offset),
                "upheld": offset % 2,
                "reference": f"DRN-{offset}",
            }
        )
    frame = pd.DataFrame(rows)
    train, val, test = time_split(frame)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    assert len(train) + len(val) + len(test) == len(frame)
    assert len(test) >= 15


def _toy_service() -> TriageService:
    credit = [
        "The customer complains the hire purchase agreement for the car was unaffordable and the lender did not check income.",
        "She complains about a personal loan taken out for home improvements and says the repayments were never affordable.",
        "He complains the credit card limit was increased without a proper look at his wages and rent.",
        "The complaint is that a catalogue credit account was opened without an affordability assessment.",
    ]
    insurance = [
        "The customer complains the insurer declined a home insurance claim after a pipe leaked in the kitchen.",
        "She complains the motor insurer refused the claim because it said she had not disclosed a previous accident.",
        "He complains the travel insurer declined a cancelled holiday after a medical emergency.",
        "The complaint is that the pet insurer stopped paying for treatment it had previously covered.",
    ]
    texts = credit + insurance
    labels = ["Consumer Credit"] * 4 + ["Insurance"] * 4
    upheld = [1, 1, 0, 1, 0, 0, 1, 0]
    product = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=400)),
        ]
    )
    product.fit(texts, labels)
    uphold_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=400)),
        ]
    )
    uphold_pipe.fit(texts, upheld)
    scores = uphold_pipe.decision_function(texts)
    calibrator = PlattCalibrator().fit(scores, upheld)
    bundle = UpholdBundle(pipeline=uphold_pipe, calibrator=calibrator, threshold=0.5)
    return TriageService(
        product,
        bundle,
        {"uphold_threshold": 0.5, "priority_cuts": {"high": 0.7}, "disclaimer": "test"},
    )


def test_triage_endpoint():
    service = _toy_service()
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    response = client.post(
        "/triage",
        json={
            "text": "The customer complains the insurer declined a buildings claim after a pipe leaked and ruined the kitchen units.",
            "business_name": "Example Insurance Ltd",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["product"]["label"] == "Insurance"
    assert 0 <= body["uphold"]["probability"] <= 1
    assert body["uphold"]["priority"] in {"high", "medium", "standard"}
    assert body["summary"]
    assert body["firm_type"] == "Insurer"
    assert "DISP" in body["disclaimer"] or body["disclaimer"] == "test"
    too_short = client.post("/triage", json={"text": "too short"})
    assert too_short.status_code == 422
    app.dependency_overrides.clear()


def test_full_decision_is_stripped_before_scoring():
    service = _toy_service()
    pasted = (
        "The complaint\n\n"
        "The customer complains the insurer declined a buildings claim after a pipe leaked.\n\n"
        "What happened\n\n"
        "A pipe leaked and ruined the kitchen.\n\n"
        "What I've decided - and why\n\n"
        "I've decided to uphold the complaint and award £500.\n"
    )
    result = service.triage(pasted)
    assert result["intake_text_used"] is True
    assert "£500" not in result["summary"]
