"""Límites de países (Natural Earth 1:50m, vía el paquete world-atlas).

La API de FIRMS devuelve un rectángulo, pero el archivo histórico viene recortado
por país. Para que los dos se puedan sumar, los hotspots de la API se recortan con
el contorno del país.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

TOPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geo", "countries-50m.json")
# nombres de archivo de FIRMS que Natural Earth escribe distinto
ALIASES = {
    "United States": "United States of America",
    "Democratic Republic of the Congo": "Dem. Rep. Congo",
    "Republic of the Congo": "Congo",
    "Central African Republic": "Central African Rep.",
    "South Sudan": "S. Sudan",
    "Dominican Republic": "Dominican Rep.",
    "Bosnia and Herzegovina": "Bosnia and Herz.",
    "Ivory Coast": "Côte d'Ivoire",
    "Equatorial Guinea": "Eq. Guinea",
    "Solomon Islands": "Solomon Is.",
}


def _decode_arcs(topo: dict) -> list[np.ndarray]:
    """Arcos de TopoJSON en longitud y latitud (deshace la cuantización delta)."""
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        q = np.cumsum(np.asarray(arc, dtype=float), axis=0)
        arcs.append(np.column_stack([q[:, 0] * sx + tx, q[:, 1] * sy + ty]))
    return arcs


def _ring(indices: list[int], arcs: list[np.ndarray]) -> np.ndarray:
    parts = []
    for i in indices:
        a = arcs[i] if i >= 0 else arcs[~i][::-1]
        parts.append(a if not parts else a[1:])
    return np.vstack(parts)


@lru_cache(maxsize=1)
def _load() -> tuple[dict, list[np.ndarray]]:
    with open(TOPO, encoding="utf-8") as fh:
        topo = json.load(fh)
    return topo, _decode_arcs(topo)


def country_rings(name: str) -> list[np.ndarray]:
    """Anillos (exteriores y huecos) del país. Lanza KeyError si no lo encuentra."""
    topo, arcs = _load()
    wanted = ALIASES.get(name.replace("_", " "), name.replace("_", " "))
    for g in topo["objects"]["countries"]["geometries"]:
        if g["properties"].get("name") != wanted:
            continue
        polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
        return [_ring(r, arcs) for poly in polys for r in poly]
    raise KeyError(f"no encuentro '{name}' en los límites de Natural Earth")


def inside(lon: np.ndarray, lat: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Punto en polígono por paridad de cruces; los huecos se restan solos."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    result = np.zeros(lon.shape, dtype=bool)
    for ring in rings:
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        cand = np.flatnonzero((lon >= lo[0]) & (lon <= hi[0]) & (lat >= lo[1]) & (lat <= hi[1]))
        if cand.size == 0:
            continue
        x, y = lon[cand], lat[cand]
        x1, y1 = ring[:-1, 0], ring[:-1, 1]
        x2, y2 = ring[1:, 0], ring[1:, 1]
        crosses = np.zeros(cand.size, dtype=bool)
        for k in range(x1.size):
            if y1[k] == y2[k]:
                continue
            hit = ((y1[k] > y) != (y2[k] > y)) & (
                x < x1[k] + (y - y1[k]) * (x2[k] - x1[k]) / (y2[k] - y1[k]))
            crosses ^= hit
        result[cand] ^= crosses
    return result


class CountryMask:
    """Contorno del país rasterizado a `res` grados y ensanchado `buffer_km`.

    El margen absorbe la diferencia entre los límites de Natural Earth 1:50m y los
    que usa FIRMS para cortar su archivo, que siguen ríos con más detalle.
    """

    def __init__(self, name: str, res: float = 0.01, buffer_km: float = 2.0):
        rings = country_rings(name)
        allpts = np.vstack(rings)
        pad = buffer_km / 100.0 + 2 * res
        self.lon0, self.lat0 = allpts[:, 0].min() - pad, allpts[:, 1].min() - pad
        self.res = res
        nx = int(np.ceil((allpts[:, 0].max() + pad - self.lon0) / res))
        ny = int(np.ceil((allpts[:, 1].max() + pad - self.lat0) / res))
        cx = self.lon0 + (np.arange(nx) + 0.5) * res
        cy = self.lat0 + (np.arange(ny) + 0.5) * res
        gx, gy = np.meshgrid(cx, cy)
        grid = inside(gx.ravel(), gy.ravel(), rings).reshape(ny, nx)
        cells = int(np.ceil(buffer_km / (111.32 * res)))
        out = grid.copy()
        for dy in range(-cells, cells + 1):
            for dx in range(-cells, cells + 1):
                if dx * dx + dy * dy <= cells * cells:
                    out |= np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        self.grid = out
        self.bbox = (self.lon0, self.lat0, self.lon0 + nx * res, self.lat0 + ny * res)

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        ix = np.floor((np.asarray(lon) - self.lon0) / self.res).astype(int)
        iy = np.floor((np.asarray(lat) - self.lat0) / self.res).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < self.grid.shape[1]) & (iy < self.grid.shape[0])
        out = np.zeros(ix.shape, dtype=bool)
        out[ok] = self.grid[iy[ok], ix[ok]]
        return out
