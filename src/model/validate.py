"""Validation: compare predicted alert streets against ground truth."""
import numpy as np


def f1_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    y_pred = scores >= threshold
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def validate(scored: list[dict], truth_ids: set[str], threshold: float = 70.0):
    ids = [s["id"] for s in scored]
    y_true = np.array([1 if i in truth_ids else 0 for i in ids])
    scores = np.array([s["risk"] for s in scored])
    precision, recall, f1 = f1_at_threshold(y_true, scores, threshold)
    return dict(precision=round(precision, 3), recall=round(recall, 3),
                f1=round(f1, 3), threshold=threshold,
                n_alert=int((scores >= threshold).sum()),
                n_truth=len(truth_ids))
