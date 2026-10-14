# FloodWatch 🌊

**Live demo:** **[floodwatch.streamlit.app](https://floodwatch.streamlit.app)** *(auto-deploys from main)*

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://floodwatch.streamlit.app)
[![CI](https://github.com/Sreecharan-lagudu/Floodwatch/actions/workflows/streamlit.yml/badge.svg)](https://github.com/Sreecharan-lagudu/Floodwatch/actions/workflows/streamlit.yml)

**Street-level flood alerts for Indian cities** — Smart India Hackathon 2026 problem SIH26085 (Ministry of Earth Sciences), built as a CurveBall squad project.

> Monsoon forecasts tell you *how much* rain falls — not *which street goes under*. FloodWatch couples rainfall nowcasts with terrain and drainage data to predict flooding street by street, and pushes alerts before the water arrives.

**Demo coverage:** 16 flood-prone cities across Indian states (Chennai, Mumbai, Delhi, Kolkata, Hyderabad, Ahmedabad, Bengaluru, Guwahati, Patna, Lucknow, Bhopal, Kochi, Visakhapatnam, Cuttack, Srinagar, Dehradun) — selectable in the dashboard, each with its own terrain model (coastal / river / delta / inland) and rainfall profile.

## How it works

```
Rainfall nowcast (3-6h) ─┐
DEM: elevation + flow    │
accumulation             ├──▶ Risk engine ──▶ street-segment
Street network (OSM)     │    (weighted       flood-risk score 0-100
Drainage proxy layer    ─┘     rule-based)         │
                                                ├──▶ Streamlit + Folium map (live)
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

## Live demo

The dashboard is deployed on **Streamlit Community Cloud**:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with **GitHub**
2. **New app** → repo: `Sreecharan-lagudu/Floodwatch` → branch: `main` → main file: `src/app/dashboard.py`
3. (Recommended) set custom subdomain: `floodwatch` → **Deploy**

First deploy builds a fresh environment from `requirements.txt` (~5 min). After that, **every push to `main` auto-redeploys** — the app stays live with your latest code.

**Interview tip:** open the live link, drag the **Forecast hour** slider to H+4 (peak cloudburst), and let the interviewer watch streets flip green → orange → red in real time, with the mock SMS panel firing. Then show the per-city validation table from `run_demo.py` (precision/recall/F1 against flood ground truth per city).

## Quickstart (local)

```bash
pip install -r requirements.txt
python run_demo.py            # builds demo cities + runs the full pipeline, all 16 states
streamlit run src/app/dashboard.py   # open the live map
```

## Real data mode

Point the pipeline at real city data:

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
│   └── synthetic/     # demo city generators (16 states)
├── src/
│   ├── ingestion/     # fetch_osm.py, process_dem.py, fetch_imd.py
│   ├── features/      # terrain.py, drainage_proxy.py, rainfall.py
│   ├── model/         # risk_engine.py, calibration.py, validate.py
│   └── app/           # dashboard.py (Streamlit+Folium), alerts.py
├── .streamlit/config.toml   # app metadata for Streamlit Cloud
├── run_demo.py        # end-to-end synthetic pipeline, all cities
└── requirements.txt
```

## Validation approach

The risk engine is calibrated so that streets documented as flooded in the Dec 2015
event score above the alert threshold, reported with precision/recall/F1 per city
(see `src/model/validate.py` and `run_demo.py` output). On the synthetic demo the
generator embeds known flood-prone streets, so every run reports honest metrics.

## Honest limits

- Municipal drainage network data is not public — we use proxies (distance to
  waterway, land cover). Operational deployment needs city corporation data.
- "Near-real-time" is simulated by replaying rainfall event timelines.
- This is a squad demo for SIH26085, not an operational early-warning system.

## Squad

- Data science: [Sree Charan Lagudu](https://github.com/Sreecharan-lagudu)
- GIS: *open* — see [issue #1](https://github.com/Sreecharan-lagudu/Floodwatch/issues/1)
- Developer: *open* — see [issue #2](https://github.com/Sreecharan-lagudu/Floodwatch/issues/2)

## License

MIT
