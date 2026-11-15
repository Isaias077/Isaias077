#!/usr/bin/env python3
"""Genera server/geo/regions.json.gz con países y provincias simplificados.

Fuente: Natural Earth 1:10m (dominio público), tal como viene en el paquete npm
geojson-places@1.0.8 (licencia ISC):

    npm pack geojson-places@1.0.8 && tar -xzf geojson-places-1.0.8.tgz
    python tools/build_regions.py package/data

Cada país es la unión de sus provincias, así los dos niveles calzan exactos.
"""
import gzip
import json
import os
import sys
from collections import defaultdict

import shapely
from shapely.geometry import mapping, shape

TOL_PROVINCE = 0.01     # ~1 km
TOL_COUNTRY = 0.02
DIGITS = 3              # ~100 m


def rounded(geom, tol):
    g = shapely.simplify(geom, tol, preserve_topology=True)
    g = shapely.set_precision(g, 10 ** -DIGITS)
    return mapping(g)


def main(src: str, out: str) -> None:
    with open(os.path.join(src, "states", "admin1.json"), encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    with open(os.path.join(src, "countries", "countries.json"), encoding="utf-8") as fh:
        index = {c["country_a3"]: c for c in json.load(fh)}

    provinces, by_country, names = [], defaultdict(list), {}
    for f in feats:
        p = f["properties"]
        geom = shapely.make_valid(shape(f["geometry"]))
        iso3 = p["adm0_a3"]
        by_country[iso3].append(geom)
        names.setdefault(iso3, p["admin"])
        provinces.append({
            "id": p["adm1_code"], "iso": p.get("iso_3166_2") or "", "pais": iso3,
            "nombre": p.get("name_es") or p.get("name") or p["adm1_code"],
            "tipo": p.get("type_en") or "",
            "bbox": [round(v, 3) for v in geom.bounds],
            "geom": rounded(geom, TOL_PROVINCE),
        })
    # nombres repetidos dentro de un país (p. ej. los dos "Buenos Aires"): nombre local o código
    seen = defaultdict(list)
    for p, f in zip(provinces, feats):
        seen[(p["pais"], p["nombre"])].append((p, f["properties"]))
    for group in seen.values():
        if len(group) > 1:
            for p, props in group:
                p["nombre"] = props.get("name") or p["nombre"]
    for key in ("tipo", "id"):                 # luego el tipo (ciudad, condado...) y, si no, el código
        seen = defaultdict(list)
        for p in provinces:
            seen[(p["pais"], p["nombre"])].append(p)
        for group in seen.values():
            if len(group) > 1 and len({p[key] for p in group}) == len(group):
                for p in group:
                    p["nombre"] = f"{p['nombre']} ({p[key]})"

    countries = []
    for iso3, geoms in by_country.items():
        geom = shapely.union_all(geoms)
        info = index.get(iso3, {})
        countries.append({
            "iso3": iso3, "iso2": info.get("country_a2", ""),
            "nombre": info.get("i18n", {}).get("es") or names[iso3],
            "nombre_en": names[iso3],
            "bbox": [round(v, 3) for v in geom.bounds],
            "geom": rounded(geom, TOL_COUNTRY),
        })
    countries.sort(key=lambda c: c["nombre"])
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump({"fuente": "Natural Earth 1:10m vía geojson-places 1.0.8",
                   "paises": countries, "provincias": provinces}, fh, separators=(",", ":"))
    print(f"{len(countries)} países y {len(provinces)} provincias en {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1], os.path.join(here, "server", "geo", "regions.json.gz"))
