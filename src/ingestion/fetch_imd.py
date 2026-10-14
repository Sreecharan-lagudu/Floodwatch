"""Real-data mode: pull IMD rainfall observations/nowcast into a simple CSV.

IMD gridded data and city nowcasts are distributed via imd.gov and data.gov.in.
Because endpoints and formats change, this module documents the expected
schema and offers a stub loader; wire it to the current IMD feed when deploying.

Expected CSV columns: station,timestamp,rain_mm
"""
import argparse
import csv


def load_rainfall_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(timestamp=r["timestamp"],
                             station=r["station"],
                             rain_mm=float(r["rain_mm"])))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="local IMD export CSV (station,timestamp,rain_mm)")
    p.add_argument("--out", default="data/raw/rainfall.csv")
    args = p.parse_args()
    if not args.csv:
        print("Download the IMD gridded/nowcast product from imd.gov (or data.gov.in),")
        print("export to CSV with columns station,timestamp,rain_mm, then re-run:")
        print("  python -m src.ingestion.fetch_imd --csv <file>")
        return
    rows = load_rainfall_csv(args.csv)
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=["station", "timestamp", "rain_mm"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
