"""FloodWatch demo dashboard (Streamlit + Folium).

Run:
    streamlit run src/app/dashboard.py
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np

from src.model.risk_engine import score_streets, DEFAULT_WEIGHTS
from src.app.alerts import generate_alerts, render_sms

st.set_page_config(page_title="FloodWatch — India", layout="wide")
st.title("🌊 FloodWatch — street-level flood alerts")
st.caption("SIH26085 squad demo · rainfall + terrain + drainage proxies → per-street risk")

CITIES_FILE = "data/processed/cities.json"
LEGACY_FILE = "data/processed/demo_city.json"


@st.cache_data
def load_all():
    if os.path.exists(CITIES_FILE):
        with open(CITIES_FILE) as f:
            return json.load(f)
    if os.path.exists(LEGACY_FILE):
        with open(LEGACY_FILE) as f:
            single = json.load(f)
        single["meta"]["state"] = "Tamil Nadu"
        return {"chennai": single}
    from data.synthetic.generate_cities import main as gen
    gen()
    with open(CITIES_FILE) as f:
        return json.load(f)


@st.cache_data
def load_features(city):
    from src.pipeline import build_static_features
    return build_static_features(city)


cities = load_all()

with st.sidebar:
    st.header("City")
    city_keys = list(cities.keys())
    labels = {k: f"{cities[k]['meta']['name']} · {cities[k]['meta']['state']}" for k in city_keys}
    chosen = st.select_slider("Select city (across Indian states)",
                              options=city_keys,
                              format_func=lambda k: labels[k],
                              value="chennai" if "chennai" in city_keys else city_keys[0])
    city = cities[chosen]
    feat = load_features(city)
    streets = city["streets"]
    rainfall = city["rainfall"]
    id_to_idx = {s["id"]: i for i, s in enumerate(streets)}

    st.header("Nowcast replay")
    hours = [h["hour"] for h in rainfall]
    hour = st.select_slider("Forecast hour", hours, value=hours[0])
    rain_now = next(h["rain_mm_per_h"] for h in rainfall if h["hour"] == hour)
    st.metric("Predicted rainfall", f"{rain_now:.0f} mm/h")
    w_rain = st.slider("w1 rainfall", 0.0, 1.0, DEFAULT_WEIGHTS["rainfall"], 0.05)
    w_elev = st.slider("w2 elevation", 0.0, 1.0, DEFAULT_WEIGHTS["elevation"], 0.05)
    w_flow = st.slider("w3 flow accumulation", 0.0, 1.0, DEFAULT_WEIGHTS["flow_acc"], 0.05)
    w_drain = st.slider("w4 drainage deficit", 0.0, 1.0, DEFAULT_WEIGHTS["drainage"], 0.05)
    w_hist = st.slider("w5 flood history", 0.0, 1.0, DEFAULT_WEIGHTS["history"], 0.05)

st.subheader(f"{city['meta']['name']}, {city['meta']['state']}")

weights = dict(rainfall=w_rain, elevation=w_elev, flow_acc=w_flow,
               drainage=w_drain, history=w_hist)
scored = score_streets(streets, feat, rain_now, weights)

c1, c2, c3 = st.columns(3)
alerts = [s for s in scored if s["band"] == "alert"]
watch = [s for s in scored if s["band"] == "watch"]
c1.metric("🔴 Alert streets", len(alerts))
c2.metric("🟠 Watch streets", len(watch))
c3.metric("🟢 Low risk", len(scored) - len(alerts) - len(watch))

# Esri World Street Map tiles: free, no API key, and allows cloud/CI traffic
# (OpenStreetMap tile servers block Codespaces/cloud IPs by usage policy).
ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
m = folium.Map(location=[city["meta"]["lat0"] + 0.02, city["meta"]["lon0"] + 0.03],
               zoom_start=13, tiles=ESRI_TILES, attr="Esri")
for s in scored:
    folium.PolyLine(
        [s["a"], s["b"]],
        color=s["color"], weight=6 if s["band"] == "alert" else 4, opacity=0.85,
        tooltip=f"{s['id']} — risk {s['risk']}/100 ({s['band']})",
    ).add_to(m)
st_folium(m, height=520, use_container_width=True)

st.subheader("Mock SMS / push alerts")
st.code(render_sms(generate_alerts(scored)), language="text")

st.subheader("Risk table (top 10)")
top = sorted(scored, key=lambda s: -s["risk"])[:10]
st.dataframe(
    [{"street": s["id"], "risk": s["risk"], "band": s["band"],
      "elev_score": round(float(feat["elevation"][id_to_idx[s["id"]]]), 2),
      "drain_score": round(float(feat["drainage"][id_to_idx[s["id"]]]), 2),
      "mid": ", ".join(f"{c:.4f}" for c in s["mid"])} for s in top],
    use_container_width=True,
)
st.caption("Drainage is a proxy (distance to waterway + built-up share) — municipal "
           "network data is not public. Demo cities are synthetic; real mode ingests "
           "IMD rainfall, CartoDEM and OSM.")
