"""Evaluation metrics for vulnerability detection."""
from __future__ import annotations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(preds: list[int], labels: list[int]) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
