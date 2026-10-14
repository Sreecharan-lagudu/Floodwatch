"""Rainfall intensity scoring.

Compares predicted rainfall against a flood-trigger threshold derived from
historical events (Chennai Dec 2015 style: sustained >60 mm/h cloudbursts
overwhelmed the city). Returns a per-hour score in [0, 1].
"""

# Rainfall (mm/h over the next nowcast window) at which historically
# flooding starts on already-saturated low-lying terrain.
FLOOD_RAIN_THRESHOLD_MM = 45.0


def rainfall_score(rain_mm_per_h: float, threshold: float = FLOOD_RAIN_THRESHOLD_MM) -> float:
    """Saturating score: half credit at the antecedent threshold, 1.0 at 2x."""
    import numpy as np
    if rain_mm_per_h <= 0:
        return 0.0
    return float(np.clip(rain_mm_per_h / (2 * threshold), 0, 1))
