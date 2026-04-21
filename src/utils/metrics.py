"""
src/utils/metrics.py
Evaluation metrics for FGVD classification — mirrors Tables II–V in the paper.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


CLASS_NAMES = ["car", "motorcycle", "scooter", "truck", "autorickshaw", "bus"]


def compute_metrics(y_true: list[int], y_pred: list[int], class_names=CLASS_NAMES) -> dict:
    """Return a dict of common classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                confusion_matrix=cm, report=report)


def binary_accuracy(y_true: list[int], y_pred: list[int],
                    positive_classes: list[int]) -> float:
    """
    Accuracy for one-vs-rest binary case (e.g. two-wheelers vs all).
    positive_classes: class ids considered as the positive set.
    """
    y_true_bin = [1 if y in positive_classes else 0 for y in y_true]
    y_pred_bin = [1 if y in positive_classes else 0 for y in y_pred]
    return accuracy_score(y_true_bin, y_pred_bin)


def print_summary(metrics: dict):
    print(f"\n{'='*50}")
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall   : {metrics['recall']*100:.2f}%")
    print(f"  F1       : {metrics['f1']*100:.2f}%")
    print(f"{'='*50}")
    print(metrics["report"])
