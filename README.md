# FloodWatch 🌊

**Street-level flood alerts for Chennai** — Smart India Hackathon 2026 problem SIH26085 (Ministry of Earth Sciences), built as a CurveBall squad project.

> Monsoon forecasts tell you *how much* rain falls — not *which street goes under*. FloodWatch couples rainfall nowcasts with terrain and drainage data to predict flooding street by street, and pushes alerts before the water arrives.

## How it works

```
Rainfall nowcast (3-6h) ─┐
DEM: elevation + flow    │
accumulation             ├──▶ Risk engine ──▶ street-segment
Street network (OSM)     │    (weighted       flood-risk score 0-100
Drainage proxy layer    ─┘     rule-based)         │
                                                ├──▶ Streamlit + Folium map
                                                └──▶ Alerts (SMS/push mock)
```

Each street segment gets a transparent, explainable score:

```
risk(s) = w1·rainfall_intensity_score
        + w2·low_elevation_score
        + w3·flow_accumulation_score
        + w4·drainage_deficit_score
        + w5·flood_history_score
```

Risk bands: **0-39 low · 40-69 watch · 70-100 alert**.

We deliberately chose a rule-based, calibrated model over a black-box ML model: it needs no training corpus, every alert is explainable to citizens and judges, and weights are tuned against the December 2015 Chennai flood ground truth.

## Quickstart (synthetic demo, zero downloads)

```bash
pip install -r requirements.txt
python run_demo.py            # builds a synthetic city + runs the full pipeline
streamlit run src/app/dashboard.py   # open the live map
```

The synthetic demo generates a small coastal city with a river, low-lying wards and a
rainstorm replay — so the entire pipeline (ingestion → features → risk engine →
alerts) runs end-to-end out of the box, no external data needed.

## Real data mode

Point the pipeline at real Chennai data:

| Layer | Source |
|---|---|
| Rainfall nowcast | IMD gridded rain gauge data (imd.gov / data.gov.in) |
| Elevation (DEM) | CartoDEM 30m (Bhuvan) or Copernicus 30m |
| Street network | OpenStreetMap via OSMnx |
| Drainage proxy | distance to waterway + land-cover built-up % (drainage network is not public) |
| Ground truth | Dec 2015 Chennai flood reports; Sentinel-1 flood extents (Copernicus) |

```bash
python -m src.ingestion.fetch_osm --city "Chennai, India" --out data/processed/streets.json
python -m src.ingestion.process_dem --dem data/raw/dem_chennai.tif --out data/processed/terrain.json
python -m src.ingestion.fetch_imd --out data/raw/rainfall.csv
python -m src.model.calibrate --truth data/raw/flood2015.csv
```

## Project structure

```
floodwatch/
├── data/
│   ├── raw/           # downloads (gitignored)
│   ├── processed/     # pipeline outputs (GeoJSON/JSON)
│   └── synthetic/     # demo city generator
├── src/
│   ├── ingestion/     # fetch_osm.py, process_dem.py, fetch_imd.py
│   ├── features/      # terrain.py, drainage_proxy.py, rainfall.py
│   ├── model/         # risk_engine.py, calibration.py, validate.py
│   └── app/           # dashboard.py (Streamlit+Folium), alerts.py
├── run_demo.py        # end-to-end synthetic pipeline
└── requirements.txt
```

## Validation approach

The risk engine is calibrated so that streets documented as flooded in the Dec 2015
event score above the alert threshold, and reported with precision/recall (see
`src/model/validate.py`). On the synthetic demo the generator embeds known
flood-prone streets, so every run reports honest metrics.

## Honest limits

- Municipal drainage network data is not public — we use proxies (distance to
  waterway, land cover). Operational deployment needs city corporation data.
- "Near-real-time" is simulated by replaying rainfall event timelines.
- This is a squad demo for SIH26085, not an operational early-warning system.

## Squad

- Data science: [Sree Charan Lagudu](https://github.com/Sreecharan-lagudu)
- GIS: *open*
- Developer: *open*

## License

MIT
