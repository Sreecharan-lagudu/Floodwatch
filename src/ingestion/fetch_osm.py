"""Real-data mode: fetch a city street network from OpenStreetMap via OSMnx.

Usage:
    python -m src.ingestion.fetch_osm --city "Chennai, India" --out data/processed/streets.json
"""
import argparse
import json


def main():
    import osmnx as ox  # only required in real-data mode

    p = argparse.ArgumentParser()
    p.add_argument("--city", default="Chennai, India")
    p.add_argument("--out", default="data/processed/streets.json")
    args = p.parse_args()

    G = ox.graph_from_place(args.city, network_type="drive")
    streets = []
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        streets.append(dict(
            id=f"S{i:05d}",
            a=[G.nodes[u]["y"], G.nodes[u]["x"]],
            b=[G.nodes[v]["y"], G.nodes[v]["x"]],
            mid=[(G.nodes[u]["y"] + G.nodes[v]["y"]) / 2,
                 (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2],
            name=data.get("name", ""),
            length_m=data.get("length"),
            builtup=0.6,  # replace with land-cover layer when available
        ))
    with open(args.out, "w") as f:
        json.dump(streets, f)
    print(f"wrote {args.out}: {len(streets)} street segments")


if __name__ == "__main__":
    main()
