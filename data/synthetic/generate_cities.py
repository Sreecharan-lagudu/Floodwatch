"""Generate synthetic demo cities for FloodWatch across Indian states.

Each city gets its own deterministic synthetic street grid, elevation surface
(coastal/river/inland variants), rainfall replay and flood ground truth.
Usage:
    python -m data.synthetic.generate_cities            # writes data/processed/cities.json
"""
import json
import os
import numpy as np

SEED = 42
N_COLS, N_ROWS = 14, 10
DLAT, DLON = 0.0045, 0.0055
GRID = 60

# Flood-prone Indian cities across states (lat, lon, terrain type)
CITIES = {
    "chennai":    dict(name="Chennai",    state="Tamil Nadu",        lat=13.06, lon=80.22, terrain="coastal"),
    "mumbai":     dict(name="Mumbai",     state="Maharashtra",       lat=19.07, lon=72.87, terrain="coastal"),
    "delhi":      dict(name="Delhi",      state="Delhi",             lat=28.61, lon=77.21, terrain="river"),
    "kolkata":    dict(name="Kolkata",    state="West Bengal",       lat=22.57, lon=88.36, terrain="delta"),
    "hyderabad":  dict(name="Hyderabad",   state="Telangana",         lat=17.38, lon=78.48, terrain="inland"),
    "ahmedabad":  dict(name="Ahmedabad",  state="Gujarat",           lat=23.02, lon=72.57, terrain="river"),
    "bengaluru":  dict(name="Bengaluru",  state="Karnataka",         lat=12.97, lon=77.59, terrain="inland"),
    "guwahati":   dict(name="Guwahati",   state="Assam",             lat=26.14, lon=91.73, terrain="river"),
    "patna":      dict(name="Patna",      state="Bihar",             lat=25.59, lon=85.14, terrain="river"),
    "lucknow":    dict(name="Lucknow",    state="Uttar Pradesh",     lat=26.85, lon=80.95, terrain="inland"),
    "bhopal":     dict(name="Bhopal",    state="Madhya Pradesh",     lat=23.26, lon=77.41, terrain="inland"),
    "kochi":      dict(name="Kochi",      state="Kerala",            lat=9.93,  lon=76.27, terrain="coastal"),
    "vizag":      dict(name="Visakhapatnam", state="Andhra Pradesh", lat=17.69, lon=83.22, terrain="coastal"),
    "cuttack":    dict(name="Cuttack",    state="Odisha",            lat=20.46, lon=85.88, terrain="delta"),
    "srinagar":   dict(name="Srinagar",   state="Jammu & Kashmir",   lat=34.08, lon=74.80, terrain="river"),
    "dehradun":   dict(name="Dehradun",   state="Uttarakhand",       lat=30.32, lon=78.03, terrain="inland"),
}

def elevation_surface(kind: str, rng) -> np.ndarray:
    y, x = np.mgrid[0:GRID, 0:GRID].astype(float)
    if kind == "coastal":
        elev = 18.0 * (1.0 - x / GRID) + 6.0 * (1.0 - y / GRID)
        river_x = 0.45 * GRID + 4.0 * np.sin(2 * np.pi * y / GRID)
        elev -= 9.0 * np.exp(-((x - river_x) ** 2) / (2 * 3.0 ** 2))
    elif kind in ("river", "delta"):
        elev = 14.0 * (1.0 - y / GRID) + 3.0 * np.abs(x / GRID - 0.5)
        river_x = 0.50 * GRID + 5.0 * np.sin(2 * np.pi * y / GRID)
        elev -= 11.0 * np.exp(-((x - river_x) ** 2) / (2 * 4.0 ** 2))
        if kind == "delta":
            elev -= 6.0 * (x / GRID)  # deltas also fall toward the sea
    else:  # inland
        elev = 20.0 * (1.0 - y / GRID) + 8.0 * np.sin(x / GRID * np.pi)
        river_x = 0.40 * GRID + 3.0 * np.cos(2 * np.pi * y / GRID)
        elev -= 7.0 * np.exp(-((x - river_x) ** 2) / (2 * 2.5 ** 2))
    elev += rng.normal(0, 0.4, elev.shape)
    return elev, river_x

def build_city(key: str, meta: dict, seed_offset: int) -> dict:
    rng = np.random.default_rng(SEED + seed_offset)
    elev, river_x = elevation_surface(meta["terrain"], rng)
    lat0, lon0 = meta["lat"], meta["lon"]

    segs, sid = [], 0
    def e_at(r, c):
        return float(elev[int(r * (GRID - 1) / N_ROWS), int(c * (GRID - 1) / N_COLS)])
    for r in range(N_ROWS + 1):
        for c in range(N_COLS + 1):
            node = (lat0 + r * DLAT, lon0 + c * DLON)
            if c < N_COLS:
                segs.append(dict(id=f"S{sid:03d}", a=list(node),
                                 b=[node[0], lon0 + (c + 1) * DLON],
                                 elev=e_at(r, c), orient="E-W"))
                sid += 1
            if r < N_ROWS:
                segs.append(dict(id=f"S{sid:03d}", a=list(node),
                                 b=[lat0 + (r + 1) * DLAT, node[1]],
                                 elev=e_at(r, c), orient="N-S"))
                sid += 1
    for s in segs:
        s["mid"] = [(s["a"][0] + s["b"][0]) / 2, (s["a"][1] + s["b"][1]) / 2]
        dist_center = abs(s["mid"][1] - (lon0 + N_COLS * DLON / 2))
        s["builtup"] = float(np.clip(0.95 - 1.6 * dist_center, 0.3, 0.95))

    lon_of_river = lon0 + (river_x / (GRID - 1)) * N_COLS * DLON
    elevs = [x["elev"] for x in segs]
    truth = [s["id"] for s in segs
             if abs(s["mid"][1] - lon_of_river) < 2.0 * DLON
             and s["elev"] < np.percentile(elevs, 30)]

    # rainfall profile varies by terrain: deltas/coasts worse in bursts
    peak = 96.0 if meta["terrain"] in ("coastal", "delta") else 82.0
    series = [2.0, 3.0, 2.5, 22.0, 48.0, peak, peak * 0.74, 38.0, 12.0, 5.0, 2.0]

    return dict(
        meta=dict(city=key, name=meta["name"], state=meta["state"],
                  terrain=meta["terrain"], lat0=lat0, lon0=lon0,
                  n_streets=len(segs), n_truth_flooded=len(truth)),
        elevation=[[round(float(v), 2) for v in row] for row in elev],
        streets=segs,
        rainfall=[dict(hour=h, rain_mm_per_h=r) for h, r in enumerate(series)],
        ground_truth=truth,
    )

def main(out="data/processed/cities.json"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cities = {k: build_city(k, m, i) for i, (k, m) in enumerate(CITIES.items())}
    with open(out, "w") as f:
        json.dump(cities, f)
    n = sum(c["meta"]["n_streets"] for c in cities.values())
    print(f"wrote {out}: {len(cities)} cities across "
          f"{len(set(c['meta']['state'] for c in cities.values()))} states, {n} street segments")
    return out

if __name__ == "__main__":
    main()
