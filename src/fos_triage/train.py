"""Train the product coder and the uphold model, then write metrics and charts."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import f1_score as _f1

from fos_triage.config import AGGREGATES, FIGURES, PROCESSED, REPORTS
from fos_triage.figures import (
    plot_confusion,
    plot_rate_bars,
    plot_sector_volumes,
    plot_themes,
    plot_uphold_rates,
)
from fos_triage.firm_type import FIRM_TYPE_LABELS
from fos_triage.modeling import (
    assign_product_labels,
    error_examples,
    fit_product,
    fit_uphold,
    period_bounds,
    save_training_outputs,
    stratified_split,
    time_split,
)


def _load_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = json.loads((AGGREGATES / "population_counts.json").read_text(encoding="utf-8"))
    sectors = pd.DataFrame(payload["by_year_sector"])
    themes = pd.DataFrame(payload["by_year_theme"])
    return sectors, themes


def write_population_figures() -> None:
    sectors, themes = _load_counts()
    plot_sector_volumes(sectors, FIGURES / "sector_volumes.png")
    plot_uphold_rates(sectors, FIGURES / "uphold_rate_by_sector.png")
    plot_themes(themes, FIGURES / "theme_volumes.png")


def train() -> dict:
    frame = pd.read_parquet(PROCESSED / "decisions.parquet")
    frame["date"] = pd.to_datetime(frame["date"])
    labelled = assign_product_labels(frame)

    product_train, product_test = stratified_split(labelled)
    product_pipeline, product_metrics = fit_product(product_train, product_test)
    product_metrics["split"] = "stratified random 80/20 by product label"
    product_metrics["n_train"] = int(len(product_train))
    product_metrics["n_test"] = int(len(product_test))

    uphold_train, uphold_val, uphold_test = time_split(frame)
    bundle, uphold_metrics, scored = fit_uphold(uphold_train, uphold_val, uphold_test)
    uphold_metrics["split"] = "decision date, no shared dates; latest 15% test, preceding 15% validation"
    uphold_metrics["train"] = period_bounds(uphold_train)
    uphold_metrics["validation"] = period_bounds(uphold_val)
    uphold_metrics["test"] = period_bounds(uphold_test)

    # A second product model on the same date split, as a drift check.
    # Labels with no training support are dropped from this check only.
    drift = labelled.copy()
    d_train, d_val, d_test = time_split(drift)
    train_labels = set(d_train["product_label"].unique())
    d_test = d_test[d_test["product_label"].isin(train_labels)]
    if d_test["product_label"].nunique() > 1 and len(d_test) > 20:
        drift_pipe, _ = fit_product(d_train, d_test)
        drift_pred = drift_pipe.predict(d_test["model_text"])
        # Macro F1 over labels that actually occur in the later period, so a
        # product that has left the register does not count as a miss.
        product_metrics["time_split_macro_f1"] = float(
            _f1(d_test["product_label"], drift_pred, average="macro")
        )
        product_metrics["time_split_accuracy"] = float(
            (drift_pred == d_test["product_label"].to_numpy()).mean()
        )
        product_metrics["time_split_test"] = period_bounds(d_test)

    examples = error_examples(scored)
    holdout = scored.drop(columns=["model_text"]).copy()
    holdout["date"] = holdout["date"].dt.date.astype(str)
    REPORTS.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(REPORTS / "uphold_holdout.csv", index=False)
    (REPORTS / "error_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")

    audit_path = PROCESSED / "extraction_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    extra = {
        "dataset": {
            "n": int(len(frame)),
            "from": frame["date"].min().date().isoformat(),
            "to": frame["date"].max().date().isoformat(),
            "upheld_rate": float(frame["upheld"].mean()),
            "median_chars": float(frame["n_chars"].median()),
            "text_source": {str(k): int(v) for k, v in frame["text_source"].value_counts().items()},
            "products": {str(k): int(v) for k, v in frame["product"].value_counts().items()},
            "extraction": audit,
        },
        "error_examples": examples,
    }
    save_training_outputs(product_pipeline, product_metrics, bundle, uphold_metrics, extra)

    plot_confusion(
        product_metrics["confusion_matrix"],
        product_metrics["labels"],
        "Product coder, stratified holdout",
        FIGURES / "cm_product.png",
    )
    plot_confusion(
        uphold_metrics["confusion_matrix"],
        ["Not upheld", "Upheld"],
        "Uphold model, later-period holdout",
        FIGURES / "cm_uphold.png",
        cmap="Reds",
    )
    plot_rate_bars(
        frame,
        "product",
        "Uphold rate by product in the text sample",
        FIGURES / "uphold_by_product.png",
    )
    plot_rate_bars(
        frame,
        "firm_type",
        "Uphold rate by firm type in the text sample",
        FIGURES / "uphold_by_firm_type.png",
        label_map=FIRM_TYPE_LABELS,
    )
    write_population_figures()
    print(json.dumps({"product_macro_f1": product_metrics["macro_f1"], "uphold": {k: uphold_metrics[k] for k in ("roc_auc", "f1", "brier", "average_precision") if k in uphold_metrics}}, indent=2))
    return extra


def main() -> None:
    train()


if __name__ == "__main__":
    main()
