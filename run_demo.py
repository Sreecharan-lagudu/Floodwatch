"""End-to-end FloodWatch demo on the synthetic city.

    python run_demo.py

Pipeline: generate demo city -> build features -> score every rainfall hour
-> validate against ground truth -> export timeline + metrics.
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic.generate_demo import main as generate_city
from src.pipeline import build_static_features
from src.model.risk_engine import run_timeline, score_streets, DEFAULT_WEIGHTS
from src.model.validate import validate
from src.app.alerts import render_sms, generate_alerts


def main():
    print("== FloodWatch demo ==")
    generate_city()

    with open("data/processed/demo_city.json") as f:
        city = json.load(f)

    print("building terrain/drainage features...")
    feat = build_static_features(city)
    streets = city["streets"]

    print("scoring rainfall timeline...")
    timeline = run_timeline(streets, feat, city["rainfall"], DEFAULT_WEIGHTS)

    peak_hour = max(timeline, key=lambda h: sum(1 for s in timeline[h] if s["band"] == "alert"))
    scored_peak = timeline[peak_hour]
    metrics = validate(scored_peak, set(city["ground_truth"]))

    with open("data/processed/timeline.json", "w") as f:
        json.dump({str(h): s for h, s in timeline.items()}, f)

    print(f"\npeak alert hour: H+{peak_hour}")
    print(f"metrics @ alert threshold: {metrics}")
    print("\n--- mock alerts at peak hour ---")
    print(render_sms(generate_alerts(scored_peak)))

    n_alert = sum(1 for s in scored_peak if s["band"] == "alert")
    print(f"\n{n_alert} streets in alert band out of {len(streets)}.")
    print("Next: streamlit run src/app/dashboard.py")
    return metrics


if __name__ == "__main__":
    main()
