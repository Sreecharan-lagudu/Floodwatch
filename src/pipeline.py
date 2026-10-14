"""Shared pipeline glue: build per-street static feature scores from a city JSON."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import terrain as terrain_f


def build_static_features(city: dict) -> dict:
    """Per-street static scores: elevation, flow accumulation, drainage, history."""
    elev_grid = np.array(city["elevation"])
    acc = terrain_f.flow_accumulation(elev_grid)
    acc_score_grid = terrain_f.flow_accumulation_score(acc)

    # map each street to the raster cell of its midpoint
    meta = city["meta"]
    n_rows, n_cols = elev_grid.shape
    lat0, lon0 = meta["lat0"], meta["lon0"]
    segs = city["streets"]
    lats = [s["mid"][0] for s in segs]
    lons = [s["mid"][1] for s in segs]
    lat_span = max(lats) - min(lats) or 1.0
    lon_span = max(lons) - min(lons) or 1.0

    rows = [int((la - min(lats)) / lat_span * (n_rows - 1)) for la in lats]
    cols = [int((lo - min(lons)) / lon_span * (n_cols - 1)) for lo in lons]

    from src.features.drainage_proxy import drainage_deficit_score
    river_lon = float(np.mean(lons))  # synthetic river runs through the middle
    drainage = drainage_deficit_score(segs, river_lon)

    truth = set(city.get("ground_truth", []))
    history = np.array([1.0 if s["id"] in truth else 0.0 for s in segs])

    return dict(
        elevation=np.array([terrain_f.low_elevation_score(np.array([elev_grid[r, c]]))[0]
                            for r, c in zip(rows, cols)]),
        flow_acc=np.array([acc_score_grid[r, c] for r, c in zip(rows, cols)]),
        drainage=drainage,
        history=history,
    )
