"""Terrain feature scores from an elevation raster.

Scores are normalised to [0, 1] where 1 = worst (most flood-prone).
Works on the synthetic JSON raster and on rasterio arrays in real-data mode.
"""
import numpy as np


def low_elevation_score(elevations: np.ndarray) -> np.ndarray:
    """Percentile-based: lowest 30% of terrain scores highest."""
    ranks = np.argsort(np.argsort(elevations)) / (len(elevations) - 1)
    return np.clip((0.30 - ranks) / 0.30, 0, 1) * 0.6 + np.clip(1 - ranks, 0, 1) * 0.4


def flow_accumulation(elev: np.ndarray) -> np.ndarray:
    """D8-style flow accumulation on a small raster: how many upstream cells
    drain into each cell. Cheap O(n) approximation suitable for demo grids."""
    n_rows, n_cols = elev.shape
    acc = np.ones((n_rows, n_cols), dtype=float)
    order = np.argsort(elev, axis=None)[::-1]  # high -> low
    for idx in order:
        r, c = divmod(int(idx), n_cols)
        best, br, bc = None, r, c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) == (0, 0):
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols and elev[nr, nc] < elev[r, c]:
                    if best is None or elev[nr, nc] < elev[br, bc]:
                        best, br, bc = True, nr, nc
        if (br, bc) != (r, c):
            acc[br, bc] += acc[r, c]
    return acc


def flow_accumulation_score(acc: np.ndarray) -> np.ndarray:
    """Log-scaled accumulation, normalised to [0, 1]."""
    return np.clip(np.log1p(acc) / np.log1p(acc.max()), 0, 1)
