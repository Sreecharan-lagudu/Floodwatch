"""Generate a synthetic coastal city for the FloodWatch demo.

Creates:
  - a street network on a grid (id, lat/lon endpoints)
  - an elevation raster (coast on the east, river channel, inland hills)
  - a rainfall event timeline (dry start -> cloudburst -> decay)
  - ground-truth flood-prone streets (low + near-river + high rainfall)

Everything is deterministic (fixed seed) so runs are reproducible.
"""
import json
import os
import numpy as np

SEED = 42
N_COLS, N_ROWS = 14, 10          # street grid
LAT0, LON0 = 13.06, 80.22        # ~ Chennai
DLAT, DLON = 0.0045, 0.0055
GRID = 60                        # elevation raster resolution

def elevation_surface() -> np.ndarray:
    """Coastal plain: sea to the east, river valley, hills inland."""
    rng = np.random.default_rng(SEED)
    y, x = np.mgrid[0:GRID, 0:GRID].astype(float)
    # base slope: high inland (west), low at coast (east)
    elev = 18.0 * (1.0 - x / GRID) + 6.0 * (1.0 - y / GRID)
    # river valley running roughly north-south at x ~ 0.45*GRID, meandering
    river_x = 0.45 * GRID + 4.0 * np.sin(2 * np.pi * y / GRID)
    elev -= 9.0 * np.exp(-((x - river_x) ** 2) / (2 * 3.0 ** 2))
    # gentle noise
    elev += rng.normal(0, 0.4, elev.shape)
    return elev

def river_centerline() -> float:
    return 0.45 * GRID

def streets(elev: np.ndarray) -> list[dict]:
    """Grid of street segments between adjacent intersections."""
    segs = []
    sid = 0
    def e_at(i, j):
        return float(elev[int(i * (GRID - 1) / N_ROWS), int(j * (GRID - 1) / N_COLS)])
    for r in range(N_ROWS + 1):
        for c in range(N_COLS + 1):
            node = (LAT0 + r * DLAT, LON0 + c * DLON)
            if c < N_COLS:   # horizontal segment
                nxt = (LAT0 + r * DLAT, LON0 + (c + 1) * DLON)
                segs.append(dict(id=f"S{sid:03d}", a=list(node), b=list(nxt),
                                  elev=e_at(r, c), orient="E-W"))
                sid += 1
            if r < N_ROWS:   # vertical segment
                nxt = (LAT0 + (r + 1) * DLAT, LON0 + c * DLON)
                segs.append(dict(id=f"S{sid:03d}", a=list(node), b=list(nxt),
                                  elev=e_at(r, c), orient="N-S"))
                sid += 1
    for s in segs:  # midpoint + built-up % (denser near centre/coast)
        mid_lat = (s["a"][0] + s["b"][0]) / 2
        mid_lon = (s["a"][1] + s["b"][1]) / 2
        s["mid"] = [mid_lat, mid_lon]
        dist_center = abs(mid_lon - (LON0 + N_COLS * DLON / 2))
        s["builtup"] = float(np.clip(0.95 - 1.6 * dist_center, 0.3, 0.95))
    return segs

def rainfall_event() -> list[dict]:
    """Hourly nowcast timeline: 3 dry hours, 6h cloudburst, 3h decay (mm/h)."""
    dry = [2.0, 3.0, 2.5]
    burst = [22.0, 48.0, 82.0, 96.0, 71.0, 38.0]
    decay = [12.0, 5.0, 2.0]
    series = dry + burst + decay
    return [dict(hour=h, rain_mm_per_h=r) for h, r in enumerate(series)]

def ground_truth(segs: list[dict], river_x: float) -> list[str]:
    """Streets truly flood-prone: low elevation AND close to the river."""
    truth = []
    lon_of_river = LON0 + (river_x / (GRID - 1)) * N_COLS * DLON
    for s in segs:
        near_river = abs(s["mid"][1] - lon_of_river) < 2.0 * DLON
        low = s["elev"] < np.percentile([x["elev"] for x in segs], 30)
        if near_river and low:
            truth.append(s["id"])
    return truth

def main(out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    elev = elevation_surface()
    segs = streets(elev)
    truth = ground_truth(segs, river_centerline())
    payload = dict(
        meta=dict(seed=SEED, grid=GRID, lat0=LAT0, lon0=LON0,
                  n_streets=len(segs), n_truth_flooded=len(truth)),
        elevation=[[round(float(v), 2) for v in row] for row in elev],
        streets=segs,
        rainfall=rainfall_event(),
        ground_truth=truth,
    )
    path = os.path.join(out_dir, "demo_city.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"wrote {path}: {len(segs)} streets, {len(truth)} flood-prone (truth), "
          f"{len(payload['rainfall'])}h rainfall timeline")
    return path

if __name__ == "__main__":
    main()
