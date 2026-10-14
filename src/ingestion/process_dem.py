"""Real-data mode: process a DEM raster into terrain features.

Usage:
    python -m src.ingestion.process_dem --dem data/raw/dem_chennai.tif --out data/processed/terrain.json

Works with CartoDEM 30m (Bhuvan) or Copernicus GLO-30 GeoTIFFs.
"""
import argparse
import json
import numpy as np


def main():
    import rasterio
    from src.features import terrain as terrain_f

    p = argparse.ArgumentParser()
    p.add_argument("--dem", required=True)
    p.add_argument("--out", default="data/processed/terrain.json")
    args = p.parse_args()

    with rasterio.open(args.dem) as src:
        elev = src.read(1).astype(float)
        elev[elev == src.nodata] = np.nan

    # fill NaNs at edges with the median to keep D8 accumulation stable
    med = np.nanmedian(elev)
    elev = np.nan_to_num(elev, nan=med)

    acc = terrain_f.flow_accumulation(elev)
    acc_score = terrain_f.flow_accumulation_score(acc)
    low_score = terrain_f.low_elevation_score(elev.ravel()).reshape(elev.shape)

    payload = dict(
        shape=list(elev.shape),
        elevation=[[round(float(v), 2) for v in row] for row in elev[::4, ::4]],
        flow_accumulation_score=[[round(float(v), 3) for v in row] for row in acc_score[::4, ::4]],
        low_elevation_score=[[round(float(v), 3) for v in row] for row in low_score[::4, ::4]],
    )
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
