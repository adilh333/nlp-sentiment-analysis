"""Classical complaint-triage models.

Product coding uses a stratified random split: the label is a category the
register assigns, and the taxonomy is stable. Uphold prediction uses a
date split so the test period is later than training. The ombudsman's
practice drifts (motor-finance commission and scam refunds are recent
examples), and a random split would flatter the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fos_triage.config import MIN_PRODUCT_COUNT, MODELS, RANDOM_STATE, REPORTS


class PlattCalibrator:
    """Sigmoid calibration fit on a later time slice, not on the training rows."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=500)

    def fit(self, scores, y):
        self.model.fit(np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, scores) -> np.ndarray:
        return self.model.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]


@dataclass
class UpholdBundle:
    pipeline: Pipeline
    calibrator: PlattCalibrator
    threshold: float

    def decision_function(self, texts) -> np.ndarray:
        return self.pipeline.decision_function(texts)

    def predict_proba(self, texts) -> np.ndarray:
        return self.calibrator.predict(self.decision_function(texts))

    def predict(self, texts) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True,
    )


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def assign_product_labels(frame: pd.DataFrame, min_count: int = MIN_PRODUCT_COUNT) -> pd.DataFrame:
    counts = frame["product"].value_counts()
    keep = set(counts[counts >= min_count].index)
    out = frame.copy()
    out["product_label"] = out["product"].where(out["product"].isin(keep), "Other")
    other_n = int((out["product_label"] == "Other").sum())
    if 0 < other_n < min_count:
        out = out[out["product_label"] != "Other"].copy()
    return out


def time_split(frame: pd.DataFrame, test_frac: float = 0.15, val_frac: float = 0.15):
    """Split by decision date. No date is shared across train, validation and test."""
    counts = frame.groupby("date").size().sort_index()
    total = int(counts.sum())
    target_test = total * test_frac
    target_val = total * val_frac
    test_dates: list = []
    val_dates: list = []
    n_test = 0
    n_val = 0
    for date, n in reversed(list(counts.items())):
        if n_test < target_test:
            test_dates.append(date)
            n_test += int(n)
        elif n_val < target_val:
            val_dates.append(date)
            n_val += int(n)
        else:
            break
    test_set = set(test_dates)
    val_set = set(val_dates)
    test = frame[frame["date"].isin(test_set)].copy()
    val = frame[frame["date"].isin(val_set)].copy()
    train = frame[~frame["date"].isin(test_set | val_set)].copy()
    return train, val, test


def stratified_split(frame: pd.DataFrame, label: str = "product_label", test_size: float = 0.2):
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=frame[label],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _text_pipeline() -> Pipeline:
    return Pipeline([("tfidf", make_vectorizer()), ("clf", make_classifier())])


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for threshold in np.linspace(0.2, 0.8, 25):
        score = f1_score(y_true, np.asarray(proba) >= threshold, zero_division=0)
        if score > best_f:
            best_t, best_f = float(threshold), float(score)
    return best_t, best_f


def binary_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "support_positive": int(y_true.sum()),
        "support": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        metrics["average_precision"] = float(average_precision_score(y_true, proba))
    metrics["confusion_matrix"] = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    return metrics


def top_coefficients(pipeline: Pipeline, k: int = 8) -> dict:
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["clf"]
    names = np.asarray(vectorizer.get_feature_names_out())
    coef = classifier.coef_
    labels = [str(label) for label in classifier.classes_]
    out: dict[str, dict[str, list[str]]] = {}
    if coef.shape[0] == 1:
        ordered = np.argsort(coef[0])
        out[labels[-1] if len(labels) == 2 else labels[0]] = {
            "towards_positive": names[ordered[-k:]].tolist()[::-1],
            "towards_negative": names[ordered[:k]].tolist(),
        }
        return out
    for index, label in enumerate(labels):
        ordered = np.argsort(coef[index])
        out[label] = {
            "towards_class": names[ordered[-k:]].tolist()[::-1],
        }
    return out


def _multiclass_report(y_true, y_pred, labels: list[str]) -> dict:
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            n = int(matrix[i, j])
            if n:
                pairs.append({"true": true_label, "predicted": pred_label, "n": n})
    pairs.sort(key=lambda item: item["n"], reverse=True)
    return {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in labels if label in report},
        "top_confusions": pairs[:8],
        "support": {label: int(report[label]["support"]) for label in labels if label in report},
    }


def fit_product(train: pd.DataFrame, test: pd.DataFrame) -> tuple[Pipeline, dict]:
    pipeline = _text_pipeline()
    pipeline.fit(train["model_text"], train["product_label"])
    pred = pipeline.predict(test["model_text"])
    labels = sorted(set(train["product_label"]).union(test["product_label"]))
    # Keep the classifier's class order, which is what the confusion matrix should use.
    labels = [str(label) for label in pipeline.named_steps["clf"].classes_]
    metrics = _multiclass_report(test["product_label"], pred, labels)
    majority = train["product_label"].mode().iloc[0]
    baseline_pred = np.full(len(test), majority)
    baseline = _multiclass_report(test["product_label"], baseline_pred, labels)
    metrics["majority_baseline_macro_f1"] = baseline["macro_f1"]
    metrics["majority_baseline_accuracy"] = baseline["accuracy"]
    metrics["majority_class"] = str(majority)
    metrics["top_coefficients"] = top_coefficients(pipeline, k=6)
    return pipeline, metrics


def fit_uphold(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[UpholdBundle, dict, pd.DataFrame]:
    pipeline = _text_pipeline()
    pipeline.fit(train["model_text"], train["upheld"])
    calibrator = PlattCalibrator().fit(pipeline.decision_function(val["model_text"]), val["upheld"])
    val_proba = calibrator.predict(pipeline.decision_function(val["model_text"]))
    threshold, val_f1 = best_f1_threshold(val["upheld"], val_proba)
    bundle = UpholdBundle(pipeline=pipeline, calibrator=calibrator, threshold=threshold)
    test_proba = bundle.predict_proba(test["model_text"])
    metrics = binary_metrics(test["upheld"], test_proba, threshold)
    metrics["validation_f1_at_threshold"] = val_f1
    majority = int(train["upheld"].mode().iloc[0])
    majority_pred = np.full(len(test), majority)
    metrics["majority_baseline_accuracy"] = float(accuracy_score(test["upheld"], majority_pred))
    metrics["majority_baseline_f1"] = float(f1_score(test["upheld"], majority_pred, zero_division=0))
    base_rate = float(train["upheld"].mean())
    metrics["base_rate_brier"] = float(brier_score_loss(test["upheld"], np.full(len(test), base_rate)))
    metrics["train_positive_rate"] = base_rate
    metrics["top_coefficients"] = top_coefficients(pipeline, k=10)
    scored = test[["reference", "date", "product", "business", "upheld", "model_text"]].copy()
    scored["proba"] = test_proba
    scored["pred"] = (test_proba >= threshold).astype(int)
    return bundle, metrics, scored


def error_examples(scored: pd.DataFrame, k: int = 4) -> dict:
    def pack(rows: pd.DataFrame) -> list[dict]:
        examples = []
        for row in rows.itertuples(index=False):
            text = " ".join(str(row.model_text).split())
            examples.append(
                {
                    "reference": row.reference,
                    "date": row.date.date().isoformat() if hasattr(row.date, "date") else str(row.date)[:10],
                    "product": row.product,
                    "business": row.business,
                    "upheld": int(row.upheld),
                    "probability": round(float(row.proba), 3),
                    "quote": text[:280],
                }
            )
        return examples

    false_pos = scored[(scored["upheld"] == 0) & (scored["pred"] == 1)].sort_values("proba", ascending=False)
    false_neg = scored[(scored["upheld"] == 1) & (scored["pred"] == 0)].sort_values("proba", ascending=True)
    return {
        "false_positives": pack(false_pos.head(k)),
        "false_negatives": pack(false_neg.head(k)),
    }


def period_bounds(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    return {
        "n": int(len(frame)),
        "from": frame["date"].min().date().isoformat(),
        "to": frame["date"].max().date().isoformat(),
        "upheld_rate": float(frame["upheld"].mean()),
    }


def save_training_outputs(product_pipeline, product_metrics, uphold_bundle, uphold_metrics, extra: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(product_pipeline, MODELS / "product.joblib")
    joblib.dump(uphold_bundle, MODELS / "uphold.joblib")
    payload = {"product": product_metrics, "uphold": uphold_metrics, **extra}
    (REPORTS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (MODELS / "metadata.json").write_text(
        json.dumps(
            {
                "product_labels": product_metrics.get("labels", []),
                "uphold_threshold": uphold_bundle.threshold,
                "priority_cuts": {"high": 0.40, "medium": uphold_bundle.threshold},
                "disclaimer": (
                    "Scores are statistical comparisons with published ombudsman decisions. "
                    "They are not a determination under DISP and they are not a substitute "
                    "for reading the case."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
