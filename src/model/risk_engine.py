"""FloodWatch risk engine — transparent, rule-based street flood-risk score.

risk(s) = w1*rainfall + w2*elevation + w3*flow_accum + w4*drainage + w5*history

Weights are defaults; calibrate against ground truth with calibration.py.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.rainfall import rainfall_score

DEFAULT_WEIGHTS = dict(
    rainfall=0.35,
    elevation=0.20,
    flow_acc=0.15,
    drainage=0.20,
    history=0.10,
)

BANDS = [(0, 39, "low", "green"), (40, 69, "watch", "orange"), (70, 100, "alert", "red")]


def band(score: float) -> tuple[str, str]:
    for lo, hi, name, color in BANDS:
        if lo <= score <= hi:
            return name, color
    return "low", "green"


def score_streets(streets: list[dict], feat: dict[str, np.ndarray],
                  rain_mm_per_h: float, weights: dict = DEFAULT_WEIGHTS) -> list[dict]:
    """Combine per-street static feature scores with current rainfall.

    feat keys: elevation, flow_acc, drainage, history (arrays aligned to streets).
    """
    rain = rainfall_score(rain_mm_per_h)
    elev = feat["elevation"]
    flow = feat["flow_acc"]
    drain = feat["drainage"]
    hist = feat["history"]

    risk = (weights["rainfall"] * rain + weights["elevation"] * elev
            + weights["flow_acc"] * flow + weights["drainage"] * drain
            + weights["history"] * hist)
    risk = np.clip(risk, 0, 1) * 100

    out = []
    for s, r in zip(streets, risk):
        b, color = band(float(r))
        d = dict(s)
        d.update(risk=round(float(r), 1), band=b, color=color,
                 rain_score=round(rain, 3))
        out.append(d)
    return out


def run_timeline(streets: list[dict], feat: dict, rainfall: list[dict],
                 weights: dict = DEFAULT_WEIGHTS) -> dict[int, list[dict]]:
    """Score every hour of a rainfall replay; returns {hour: scored streets}."""
    return {h["hour"]: score_streets(streets, feat, h["rain_mm_per_h"], weights)
            for h in rainfall}
