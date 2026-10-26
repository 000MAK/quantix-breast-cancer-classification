#!/usr/bin/env python3
"""Train and evaluate a small, educational breast-cancer classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_malignant": float(precision_score(labels, predictions, zero_division=0)),
        "recall_malignant": float(recall_score(labels, predictions, zero_division=0)),
        "f1_malignant": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix_labels_benign_malignant": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    dataset = load_breast_cancer()
    features = dataset.data
    malignant_index = int(np.flatnonzero(dataset.target_names == "malignant")[0])
    labels = (dataset.target == malignant_index).astype(np.int64)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=args.seed,
        stratify=labels,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.20,
        random_state=args.seed,
        stratify=y_train_val,
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    metrics = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn bundled copy)",
        "positive_class": "malignant",
        "seed": args.seed,
        "split_sizes": {
            "train": int(y_train.size),
            "validation": int(y_validation.size),
            "test": int(y_test.size),
        },
        "validation": evaluate(model, x_validation, y_validation),
        "test": evaluate(model, x_test, y_test),
        "warning": "Educational demonstration only; not for clinical use.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.joblib"
    metrics_path = args.output_dir / "metrics.json"
    joblib.dump(
        {
            "model": model,
            "feature_names": dataset.feature_names.tolist(),
            "positive_class": "malignant",
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"Train / validation / test: {y_train.size} / {y_validation.size} / {y_test.size}")
    print(f"Held-out test ROC-AUC: {metrics['test']['roc_auc']:.3f}")
    print(f"Held-out malignant recall: {metrics['test']['recall_malignant']:.3f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Educational demo only; not for clinical use.")


if __name__ == "__main__":
    main()
