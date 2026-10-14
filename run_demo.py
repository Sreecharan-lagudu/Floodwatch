"""End-to-end FloodWatch demo across all demo cities.

    python run_demo.py [--city chennai]

Pipeline: generate demo cities -> build features -> score rainfall timelines
-> validate each city against ground truth -> export timelines + metrics.
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import build_static_features
from src.model.risk_engine import run_timeline, DEFAULT_WEIGHTS
from src.model.validate import validate
from src.app.alerts import render_sms, generate_alerts


def run_city(city: dict) -> dict:
    feat = build_static_features(city)
    streets = city["streets"]
    timeline = run_timeline(streets, feat, city["rainfall"], DEFAULT_WEIGHTS)
    peak_hour = max(timeline, key=lambda h: sum(1 for s in timeline[h] if s["band"] == "alert"))
    scored_peak = timeline[peak_hour]
    metrics = validate(scored_peak, set(city["ground_truth"]))
    return dict(timeline=timeline, peak_hour=peak_hour, scored_peak=scored_peak, metrics=metrics)


def main():
    args = sys.argv[1:]
    from data.synthetic.generate_cities import main as generate
    generate()

    with open("data/processed/cities.json") as f:
        cities = json.load(f)
    if args and args[0] == "--city" and len(args) > 1:
        cities = {args[1]: cities[args[1]]}

    print("== FloodWatch demo ==")
    report = {}
    for key, city in cities.items():
        r = run_city(city)
        m = r["metrics"]
        report[key] = dict(city=city["meta"]["name"], state=city["meta"]["state"],
                           peak_hour=r["peak_hour"], **m)
        print(f"{city['meta']['name']:<16} ({city['meta']['state']:<18}) "
              f"peak H+{r['peak_hour']:<2} alerts={m['n_alert']:<3} "
              f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

    # mock alerts for the first city's peak
    first = list(cities.values())[0]
    r = run_city(first)
    print(f"\n--- mock alerts at peak hour, {first['meta']['name']} ---")
    print(render_sms(generate_alerts(r["scored_peak"])))

    with open("data/processed/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nNext: streamlit run src/app/dashboard.py")
    return report


if __name__ == "__main__":
    main()
