"""Validation utilities: threshold sweep, PR/ROC metrics for the risk engine."""
import numpy as np


def _prf(pred_pos: set, truth: set) -> dict:
    tp = len(pred_pos & truth)
    fp = len(pred_pos) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn)


def validate(scored: list, truth: set, threshold: float = 70) -> dict:
    """Alert-band streets (risk >= threshold) vs ground-truth flooded streets."""
    pred = {s["id"] for s in scored if s["risk"] >= threshold}
    out = _prf(pred, truth)
    out["n_alert"] = len(pred)
    return out


def threshold_sweep(scored: list, truth: set, lows=(30, 90, 2)) -> dict:
    """Sweep alert thresholds; return per-threshold P/R/F1 and best-F1 threshold."""
    rows = []
    for t in range(lows[0], lows[1] + 1, lows[2]):
        m = validate(scored, truth, threshold=t)
        rows.append(dict(threshold=t, **{k: m[k] for k in ("precision", "recall", "f1")}))
    best = max(rows, key=lambda r: r["f1"])
    return dict(rows=rows, best_threshold=best["threshold"], best_f1=best["f1"])


def roc_auc(scored: list, truth: set) -> float:
    """AUC of risk score separating flooded vs dry streets (Mann-Whitney U)."""
    pos = np.array([s["risk"] for s in scored if s["id"] in truth])
    neg = np.array([s["risk"] for s in scored if s["id"] not in truth])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = np.mean([1.0 if p > n else 0.5 if p == n else 0.0
                 for p in pos for n in neg])
    return float(u)
