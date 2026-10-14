"""Calibrate risk-engine weights against flood ground truth.

Uses coordinate descent on the weights (softmax-parameterised so they stay
positive and sum to 1), maximising F1 of the alert band vs ground truth.
"""
import json
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.risk_engine import score_streets, DEFAULT_WEIGHTS
from src.model.validate import f1_at_threshold


def _build_static_features(city: dict) -> dict:
    from src.pipeline import build_static_features
    return build_static_features(city)


def calibrate(city_path="data/processed/demo_city.json", alert_threshold=70.0):
    with open(city_path) as f:
        city = json.load(f)
    feat = _build_static_features(city)
    streets = city["streets"]
    truth = set(city["ground_truth"])
    y_true = np.array([1 if s["id"] in truth else 0 for s in streets])
    peak_rain = max(h["rain_mm_per_h"] for h in city["rainfall"])

    keys = list(DEFAULT_WEIGHTS.keys())
    raw = np.array([DEFAULT_WEIGHTS[k] for k in keys])

    def softmax(w):
        e = np.exp(w - w.max())
        return e / e.sum()

    def f1_for(w_raw):
        w = softmax(w_raw)
        weights = dict(zip(keys, w))
        scored = score_streets(streets, feat, peak_rain, weights)
        y_pred = np.array([1 if s["risk"] >= alert_threshold else 0 for s in scored])
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return f1, weights, prec, rec

    best = (f1_for(raw)[0], softmax(raw).tolist())
    step = 0.6
    for _ in range(40):
        improved = False
        for i in range(len(raw)):
            for direction in (+1, -1):
                trial = raw.copy()
                trial[i] += direction * step
                f1 = f1_for(trial)[0]
                if f1 > best[0] + 1e-4:
                    raw = trial
                    best = (f1, softmax(trial).tolist())
                    improved = True
        if not improved:
            step *= 0.5
            if step < 0.01:
                break

    f1, weights, prec, rec = f1_for(raw)
    return dict(weights=dict(zip(keys, [round(w, 3) for w in weights])),
                f1=round(f1, 3), precision=round(prec, 3), recall=round(rec, 3))


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
