"""FloodWatch demo dashboard (Streamlit + Folium).

Run:
    streamlit run src/app/dashboard.py
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import numpy as np

from src.model.risk_engine import score_streets, DEFAULT_WEIGHTS
from src.app.alerts import generate_alerts, render_sms

st.set_page_config(page_title="FloodWatch — India", layout="wide")
st.title("🌊 FloodWatch — street-level flood alerts")
st.caption("SIH26085 squad demo · rainfall + terrain + drainage proxies → per-street risk")

CITIES_FILE = "data/processed/cities.json"
LEGACY_FILE = "data/processed/demo_city.json"
ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
ESRI_SAT = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


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

catalog = []
for key, c in cities.items():
    m = c["meta"]
    catalog.append(dict(
        key=key, name=m["name"], state=m["state"], terrain=m["terrain"],
        lat=m["lat0"], lon=m["lon0"],
        search=" ".join([key, m["name"], m["state"], m["terrain"]]).lower(),
        label=f"{m['name']} · {m['state']} ({m['terrain']})",
    ))
labels = {c["key"]: c["label"] for c in catalog}

top1, top2 = st.columns([2, 1])
with top1:
    query = st.text_input(
        "🔎 Search a city or state (e.g. 'mumbai', 'kerala', 'delta', 'guwahati')",
        placeholder="Type a city, state or terrain type…",
    ).strip().lower()
with top2:
    matches = [c for c in catalog if query in c["search"]] if query else []
    if query:
        if not matches:
            st.warning(f"No demo city matches '{query}'. Showing Chennai.")
            matches = [c for c in catalog if c["key"] == "chennai"]
        options = [c["key"] for c in matches]
        chosen = st.select_slider(
            f"{len(matches)} match(es) — pick one",
            options=options, value=options[0],
            format_func=lambda k: labels[k])
    else:
        chosen = st.select_slider(
            "Quick pick",
            options=[c["key"] for c in catalog],
            value="chennai" if "chennai" in cities else list(cities.keys())[0],
            format_func=lambda k: labels[k])

city = cities[chosen]
feat = load_features(city)
streets = city["streets"]
rainfall = city["rainfall"]
id_to_idx = {s["id"]: i for i, s in enumerate(streets)}
meta = city["meta"]

with st.sidebar:
    st.header("Nowcast replay")
    hours = [h["hour"] for h in rainfall]
    # ✨ default to the PEAK of the storm so the demo starts dramatic
    default_hour = max(hours, key=lambda h: next(r["rain_mm_per_h"] for r in rainfall if r["hour"] == h))
    hour = st.select_slider("Forecast hour", hours, value=default_hour)
    rain_now = next(h["rain_mm_per_h"] for h in rainfall if h["hour"] == hour)
    st.metric("Predicted rainfall", f"{rain_now:.0f} mm/h")
    st.progress(min(float(rain_now) / 100.0, 1.0), text=f"Storm intensity")

    st.divider()
    st.header("Risk-model weights")
    st.caption("Sum normalized to 1 automatically")
    raw = {k: st.slider(f"w{i+1} {n}", 0.0, 1.0, float(v), 0.05)
           for i, (k, n, v) in enumerate([
               ("rainfall", "rainfall", DEFAULT_WEIGHTS["rainfall"]),
               ("elevation", "elevation", DEFAULT_WEIGHTS["elevation"]),
               ("flow_acc", "flow accumulation", DEFAULT_WEIGHTS["flow_acc"]),
               ("drainage", "drainage deficit", DEFAULT_WEIGHTS["drainage"]),
               ("history", "flood history", DEFAULT_WEIGHTS["history"]),
           ])}
    total = sum(raw.values()) or 1.0
    weights = {k: v / total for k, v in raw.items()}
    st.caption(f"Effective weights: " + ", ".join(f"{k}={v:.2f}" for k, v in weights.items()))

    st.divider()
    tile_choice = st.radio("Basemap", ["Street", "Satellite"], horizontal=True)

st.subheader(f"{meta['name']}, {meta['state']} — {meta['terrain'].capitalize()} terrain")

scored = score_streets(streets, feat, rain_now, weights)

c1, c2, c3 = st.columns(3)
alerts = [s for s in scored if s["band"] == "alert"]
watch = [s for s in scored if s["band"] == "watch"]
c1.metric("🔴 Alert streets", len(alerts))
c2.metric("🟠 Watch streets", len(watch))
c3.metric("🟢 Low risk", len(scored) - len(alerts) - len(watch))

tiles = ESRI_SAT if tile_choice == "Satellite" else ESRI_TILES
m = folium.Map(location=[meta["lat0"] + 0.02, meta["lon0"] + 0.03],
               zoom_start=13, tiles=tiles, attr="Esri")
m.add_child(Fullscreen())

for s in scored:
    folium.PolyLine(
        [s["a"], s["b"]],
        color=s["color"], weight=6 if s["band"] == "alert" else 4, opacity=0.85,
        tooltip=(f"<b>{s['id']}</b><br>risk {s['risk']}/100 ({s['band']})<br>"
                 f"<a href='https://www.openstreetmap.org/#map=16/{s['mid'][0]:.5f}/{s['mid'][1]:.5f}' "
                 f"target='_blank'>📍 view on OpenStreetMap</a>"),
    ).add_to(m)

st_data = st_folium(m, height=520, use_container_width=True, returned_objects=["last_object_clicked"])

clicked = st_data.get("last_object_clicked") if st_data else None
if clicked:
    lat, lon = clicked["lat"], clicked["lng"]

    def dist_km(s):
        dy = (s["mid"][0] - lat) * 111.0
        dx = (s["mid"][1] - lon) * 111.0 * np.cos(np.radians(lat))
        return dy * dy + dx * dx

    near = min(scored, key=dist_km)
    st.success(f"📍 Selected location: **{lat:.5f}, {lon:.5f}** — nearest segment **{near['id']}**")
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Risk score", f"{near['risk']}/100")
    dc2.metric("Band", near["band"])
    dc3.metric("Elevation score", f"{float(feat['elevation'][id_to_idx[near['id']]]):.2f}")
    dc4.metric("Drainage score", f"{float(feat['drainage'][id_to_idx[near['id']]]):.2f}")
    st.caption(f"Move the Forecast-hour slider to see how this location's risk evolves through the storm.")

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
           "IMD rainfall, CartoDEM and OSM. Click any street on the map for details.")
