"""Países, provincias y rectángulos: el área de interés de cada consulta.

Los límites vienen de Natural Earth 1:10m (ver tools/build_regions.py). Los datos se
cuentan sobre una grilla global de celdas de 0,05° (unos 5 km): un área es el
conjunto de celdas cuyo centro cae dentro de ella.
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import shapely
from shapely.geometry import box, shape

GEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geo", "regions.json.gz")
CELL = 0.05
NX = int(round(360 / CELL))

# Nombre del archivo por país en FIRMS cuando difiere del de Natural Earth.
FIRMS_ALIASES = {
    "USA": "United_States", "RUS": "Russian_Federation", "TZA": "Tanzania",
    "CZE": "Czech_Republic", "SWZ": "Swaziland", "COD": "Democratic_Republic_of_the_Congo",
    "COG": "Republic_of_the_Congo", "CIV": "Cote_d'Ivoire", "LAO": "Lao_PDR",
    "KOR": "South_Korea", "PRK": "North_Korea", "MKD": "Macedonia", "GBR": "United_Kingdom",
    "BHS": "Bahamas", "GMB": "Gambia", "TLS": "East_Timor", "CPV": "Cape_Verde",
    "BRN": "Brunei", "SYR": "Syria", "IRN": "Iran", "VNM": "Vietnam", "BOL": "Bolivia",
    "VEN": "Venezuela", "MDA": "Moldova", "SRB": "Republic_of_Serbia", "MMR": "Myanmar",
}


def cell_ids(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    ix = np.floor((np.asarray(lon, dtype=np.float64) + 180.0) / CELL).astype(np.int64)
    iy = np.floor((np.asarray(lat, dtype=np.float64) + 90.0) / CELL).astype(np.int64)
    return (np.clip(iy, 0, int(180 / CELL) - 1) * NX + np.clip(ix, 0, NX - 1)).astype(np.int32)


def cell_centers(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cells, dtype=np.int64)
    return (cells % NX + 0.5) * CELL - 180.0, (cells // NX + 0.5) * CELL - 90.0


@dataclass
class Region:
    """Área de interés ya resuelta."""

    kind: str                  # "pais", "provincia" o "rectangulo"
    key: str                   # identificador estable, para cachés y nombres de archivo
    label: str
    geom: object               # shapely
    countries: list[str]       # ISO3 de los países cuyos datos hacen falta
    whole_country: str | None = None   # si es un país entero, no hace falta recortar
    bbox: tuple = field(default=())

    def __post_init__(self):
        self.bbox = tuple(round(v, 4) for v in self.geom.bounds)


class Catalog:
    def __init__(self, path: str = GEO):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.countries = {c["iso3"]: c for c in raw["paises"]}
        self.provinces = {p["id"]: p for p in raw["provincias"]}
        self.by_country: dict[str, list[dict]] = {}
        for p in raw["provincias"]:
            self.by_country.setdefault(p["pais"], []).append(p)
        for lst in self.by_country.values():
            lst.sort(key=lambda p: p["nombre"])
        self._cgeom = {k: shape(c["geom"]) for k, c in self.countries.items()}
        self._keys = list(self._cgeom)
        self._tree = shapely.STRtree([self._cgeom[k] for k in self._keys])
        self._pgeom: dict[str, object] = {}

    # ------------------------------------------------------------ geometrías
    def country_geom(self, iso3: str):
        return self._cgeom[iso3]

    def province_geom(self, pid: str):
        if pid not in self._pgeom:
            self._pgeom[pid] = shape(self.provinces[pid]["geom"])
        return self._pgeom[pid]

    def countries_touching(self, geom) -> list[str]:
        idx = self._tree.query(geom, predicate="intersects")
        return sorted(self._keys[i] for i in idx)

    # ------------------------------------------------------------ consultas
    def locate(self, lon: float, lat: float) -> dict | None:
        pt = shapely.Point(lon, lat)
        hits = self._tree.query(pt, predicate="intersects")
        if not len(hits):
            return None
        iso3 = self._keys[int(hits[0])]
        prov = next((p for p in self.by_country.get(iso3, [])
                     if p["bbox"][0] <= lon <= p["bbox"][2] and p["bbox"][1] <= lat <= p["bbox"][3]
                     and self.province_geom(p["id"]).intersects(pt)), None)
        return {"pais": iso3, "provincia": prov["id"] if prov else None}

    def resolve(self, country: str | None = None, province: str | None = None,
                bbox: tuple | None = None) -> Region:
        if province:
            if province not in self.provinces:
                raise KeyError(f"provincia desconocida: {province}")
            p = self.provinces[province]
            c = self.countries[p["pais"]]
            return Region("provincia", f"prov_{province}", f"{p['nombre']}, {c['nombre']}",
                          self.province_geom(province), [p["pais"]])
        if country:
            if country not in self.countries:
                raise KeyError(f"país desconocido: {country}")
            return Region("pais", f"pais_{country}", self.countries[country]["nombre"],
                          self._cgeom[country], [country], whole_country=country)
        if bbox:
            w, s, e, n = bbox
            geom = box(w, s, e, n)
            return Region("rectangulo", "rect_" + "_".join(f"{v:.3f}" for v in bbox),
                          f"Rectángulo {w:.2f}, {s:.2f} a {e:.2f}, {n:.2f}", geom,
                          self.countries_touching(geom))
        raise KeyError("falta el área: país, provincia o rectángulo")

    @lru_cache(maxsize=64)
    def _cells(self, key: str, wkb: bytes) -> np.ndarray:
        geom = shapely.from_wkb(wkb)
        w, s, e, n = geom.bounds
        ix = np.arange(int(np.floor((w + 180) / CELL)), int(np.floor((e + 180) / CELL)) + 1)
        iy = np.arange(int(np.floor((s + 90) / CELL)), int(np.floor((n + 90) / CELL)) + 1)
        gx, gy = np.meshgrid(ix, iy)
        cells = (gy * NX + gx).ravel().astype(np.int32)
        lon, lat = cell_centers(cells)
        inside = shapely.contains_xy(geom, lon, lat)
        out = cells[inside]
        if out.size == 0:      # área más chica que una celda: la celda de su centro
            c = geom.centroid
            out = cell_ids(np.array([c.x]), np.array([c.y]))
        return np.sort(out)

    def region_cells(self, region: Region) -> np.ndarray | None:
        """Celdas del área; None para un país entero (se usan todos sus datos)."""
        if region.whole_country:
            return None
        return self._cells(region.key, shapely.to_wkb(region.geom))

    def clip_mask(self, iso3: str, buffer_deg: float = 0.02):
        """Contorno del país ensanchado ~2 km, para recortar lo que devuelve la API."""
        geom = shapely.buffer(self._cgeom[iso3], buffer_deg)
        shapely.prepare(geom)
        return lambda lon, lat: shapely.contains_xy(geom, lon, lat)

    def firms_names(self, iso3: str) -> list[str]:
        c = self.countries[iso3]
        out = []
        for n in (FIRMS_ALIASES.get(iso3), c["nombre_en"]):
            if n:
                n = n.replace(" ", "_")
                if n not in out:
                    out.append(n)
        return out


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    return Catalog()
