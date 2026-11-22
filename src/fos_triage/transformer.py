"""Fine-tune DistilBERT on the uphold task.

Optional. Install requirements-transformer.txt first. Classical TF-IDF models
remain the ones the API serves. Weights are written under models/transformer
and are gitignored.

    python -m fos_triage.transformer
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fos_triage.config import PROCESSED, REPORTS
from fos_triage.modeling import binary_metrics, best_f1_threshold, time_split


def main() -> None:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    frame = pd.read_parquet(PROCESSED / "decisions.parquet")
    frame["date"] = pd.to_datetime(frame["date"])
    train, val, test = time_split(frame)
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class Texts(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                list(texts),
                truncation=True,
                padding=True,
                max_length=128,
            )
            self.labels = list(map(int, labels))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: torch.tensor(value[index]) for key, value in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[index])
            return item

    train_ds = Texts(train["model_text"], train["upheld"])
    val_ds = Texts(val["model_text"], val["upheld"])
    test_ds = Texts(test["model_text"], test["upheld"])
    weights = torch.tensor(
        [
            len(train) / max((train["upheld"] == 0).sum(), 1),
            len(train) / max((train["upheld"] == 1).sum(), 1),
        ],
        dtype=torch.float,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(
        output_dir="models/transformer",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=40,
        report_to=[],
        dataloader_num_workers=0,
        use_cpu=not torch.cuda.is_available(),
        seed=42,
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    val_logits = trainer.predict(val_ds).predictions
    test_logits = trainer.predict(test_ds).predictions
    val_proba = torch.softmax(torch.tensor(val_logits), dim=1).numpy()[:, 1]
    test_proba = torch.softmax(torch.tensor(test_logits), dim=1).numpy()[:, 1]
    threshold, val_f1 = best_f1_threshold(val["upheld"], val_proba)
    metrics = binary_metrics(test["upheld"], test_proba, threshold)
    metrics["validation_f1_at_threshold"] = val_f1
    metrics["model"] = model_name
    metrics["max_length"] = 128
    metrics["epochs"] = 1
    metrics["n_train"] = int(len(train))
    metrics["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    path = REPORTS / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload["transformer_uphold"] = metrics
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
