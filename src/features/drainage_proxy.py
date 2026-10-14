"""Drainage-deficit proxy.

Municipal drainage network data is not public, so we approximate drainage
deficit with (a) distance to the nearest waterway (farther = worse drainage
connection) and (b) built-up share (more impervious surface = more runoff,
less infiltration). Normalised to [0, 1], 1 = worst.
"""
import numpy as np


def distance_to_waterway_score(streets: list[dict], waterway_lon: float,
                               lon_index: int = 1) -> np.ndarray:
    dists = np.abs([s["mid"][lon_index] - waterway_lon for s in streets])
    d = dists / (dists.max() or 1.0)
    return np.clip(d, 0, 1)


def builtup_score(streets: list[dict]) -> np.ndarray:
    return np.clip(np.array([s["builtup"] for s in streets]), 0, 1)


def drainage_deficit_score(streets: list[dict], waterway_lon: float) -> np.ndarray:
    dist_s = distance_to_waterway_score(streets, waterway_lon)
    built_s = builtup_score(streets)
    return np.clip(0.6 * dist_s + 0.4 * built_s, 0, 1)
