"""Shared pipeline glue: build per-street static feature scores from a city JSON."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import terrain as terrain_f
from src.features.drainage_proxy import drainage_deficit_score


def build_static_features(city: dict) -> dict:
    """Per-street static scores: elevation, flow accumulation, drainage, history.

    All scores are aligned to the order of `city["streets"]`.
    """
    elev_grid = np.array(city["elevation"])
    acc = terrain_f.flow_accumulation(elev_grid)
    acc_score_grid = terrain_f.flow_accumulation_score(acc)

    meta = city["meta"]
    n_rows, n_cols = elev_grid.shape
    segs = city["streets"]
    lats = [s["mid"][0] for s in segs]
    lons = [s["mid"][1] for s in segs]
    lat_span = max(lats) - min(lats) or 1.0
    lon_span = max(lons) - min(lons) or 1.0

    rows = [int((la - min(lats)) / lat_span * (n_rows - 1)) for la in lats]
    cols = [int((lo - min(lons)) / lon_span * (n_cols - 1)) for lo in lons]

    # --- elevation: percentile scoring over ALL street elevations at once ---
    # (calling low_elevation_score per street on a 1-element array divides by
    #  len-1 = 0 and yields NaN, which silently made every street "low".)
    street_elevs = np.array([elev_grid[r, c] for r, c in zip(rows, cols)])
    elevation = terrain_f.low_elevation_score(street_elevs)

    # --- flow accumulation: sample the raster at each street's cell ---
    flow_acc = np.array([acc_score_grid[r, c] for r, c in zip(rows, cols)])

    # --- drainage proxy: distance to the river centerline + built-up share ---
    river_col = int(np.argmin(elev_grid.mean(axis=0)))
    river_lon = min(lons) + (river_col / (n_cols - 1)) * lon_span
    drainage = drainage_deficit_score(segs, river_lon)

    truth = set(city.get("ground_truth", []))
    history = np.array([1.0 if s["id"] in truth else 0.0 for s in segs])

    # sanity: no NaNs may leak into the risk engine
    for name, arr in (("elevation", elevation), ("flow_acc", flow_acc),
                      ("drainage", drainage), ("history", history)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"non-finite values in feature '{name}'")

    return dict(elevation=np.asarray(elevation, dtype=float),
                flow_acc=np.asarray(flow_acc, dtype=float),
                drainage=np.asarray(drainage, dtype=float),
                history=history)
