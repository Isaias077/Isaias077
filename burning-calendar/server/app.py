"""API y servidor web del calendario de actividad de quema.

    uvicorn server.app:app --port 8000

Cualquier país, provincia o rectángulo: los datos que falten se bajan solos de FIRMS
la primera vez y el calendario se completa a medida que llegan.
"""
from __future__ import annotations

import math
import os
from datetime import date
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from shapely.geometry import mapping

from . import harmonize as hz
from .data import DataManager, day_number, iso
from .regions import Region, catalog

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MAX_RECT_COUNTRIES = 6
MAX_RECT_DEG2 = 400.0

app = FastAPI(title="Calendario de actividad de quema")


@lru_cache(maxsize=1)
def manager() -> DataManager:
    return DataManager()


def num(v) -> float | None:
    """Número para JSON: NaN (meses que todavía no llegaron) va como null."""
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 1)


def region_from(pais: str | None, provincia: str | None, bbox: str | None) -> Region:
    box = None
    if bbox:
        try:
            box = tuple(float(v) for v in bbox.split(","))
            w, s, e, n = box
        except ValueError:
            raise HTTPException(400, "bbox debe ser oeste,sur,este,norte") from None
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise HTTPException(400, "rectángulo inválido: se espera oeste < este y sur < norte")
        if (e - w) * (n - s) > MAX_RECT_DEG2:
            raise HTTPException(400, "el rectángulo es demasiado grande: elegí un país o un área menor")
    try:
        region = catalog().resolve(pais, provincia, box)
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0])) from None
    if not region.countries:
        raise HTTPException(400, "el rectángulo no toca ningún país (¿está en el mar?)")
    if len(region.countries) > MAX_RECT_COUNTRIES:
        raise HTTPException(400, f"el rectángulo toca {len(region.countries)} países; "
                                 f"el máximo es {MAX_RECT_COUNTRIES}")
    return region


def parse_day(text: str) -> int:
    try:
        return day_number(date.fromisoformat(text))
    except ValueError:
        raise HTTPException(400, f"fecha inválida: {text}") from None


def month_label(m: int) -> str:
    return f"{2000 + m // 12}-{m % 12 + 1:02d}"


def country_fit(region: Region) -> tuple[float, float, float] | None:
    """Ajuste MODIS -> VIIRS de los países enteros: respaldo para áreas con pocos datos."""
    if region.whole_country:
        return None
    whole = Region("union", "union_" + "_".join(region.countries), "",
                   catalog().country_geom(region.countries[0]), region.countries, whole_country="*")
    c = manager().monthly(whole)
    h = hz.harmonize(c["month"], c["terra"], c["aqua"], c["viirs"],
                     use_viirs=c["have_viirs"], have_modis=c["have_modis"])
    f = h["fit"]
    return (f["a"], f["b"], f["sigma"]) if f["origen"] == "area" else None


def build_calendar(region: Region) -> dict:
    dm = manager()
    for iso3 in region.countries:
        dm.ensure(iso3)
    c = dm.monthly(region)
    cov = c["coverage"]
    month = c["month"]
    h = hz.harmonize(month, c["terra"], c["aqua"], c["viirs"], fallback=country_fit(region),
                     use_viirs=c["have_viirs"], have_modis=c["have_modis"])
    last_day = max(cov["modis"], cov["viirs"])
    end = date.fromisoformat(iso(last_day))
    # el mes de la última fecha con datos está incompleto salvo que lleguen a su último día
    partial_days = None if last_day < 0 or date.fromordinal(end.toordinal() + 1).day == 1 else end.day
    partial_idx = int(np.searchsorted(month, (end.year - 2000) * 12 + end.month - 1)) if partial_days else None
    if partial_idx is not None and (partial_idx >= month.size or not np.isfinite(h["harmonized"][partial_idx])):
        partial_days = partial_idx = None
    valid = np.isfinite(h["harmonized"])
    if partial_idx is not None:
        valid[partial_idx] = False
    st = hz.calendar_stats(month, h["harmonized"], valid)
    same = None
    if partial_idx is not None:
        cp = dm.monthly(region, max_dom=partial_days)
        f = h["fit"]
        hp = hz.apply(month, cp["terra"], cp["aqua"], cp["viirs"], None, f["a"], f["b"], f["sigma"],
                      use_viirs=c["have_viirs"], have_modis=c["have_modis"])
        same = hz.same_days_comparison(month, hp["harmonized"], int(month[partial_idx]))
        st["anomaly"][partial_idx], st["rank"][partial_idx] = same["anomalia"], same["percentil"]

    def origin(i):
        if not np.isfinite(h["harmonized"][i]):
            return "pendiente"
        return "VIIRS" if h["from_viirs"][i] else "MODIS convertido"

    rows = [{
        "mes": month_label(int(m)), "anio": 2000 + int(m) // 12, "mes_del_anio": int(m) % 12 + 1,
        "armonizado": num(h["harmonized"][i]), "bajo": num(h["low"][i]), "alto": num(h["high"][i]),
        "origen": origin(i),
        "modis_terra": int(c["terra"][i]), "modis_aqua": int(c["aqua"][i]),
        "viirs": int(c["viirs"][i]) if h["from_viirs"][i] else None,
        "anomalia": None if not np.isfinite(h["harmonized"][i]) else round(float(st["anomaly"][i]), 3),
        "percentil": None if not np.isfinite(h["harmonized"][i]) else round(float(st["rank"][i]), 1),
        "parcial": i == partial_idx,
    } for i, m in enumerate(month)]

    years = sorted({r["anio"] for r in rows})
    annual = []
    for y in years:
        ys = [r for r in rows if r["anio"] == y]
        vals = [r["armonizado"] for r in ys if r["armonizado"] is not None]
        annual.append({"anio": y, "armonizado": round(sum(vals), 1) if vals else None,
                       "completo": len(vals) == 12 and not any(r["parcial"] for r in ys)})
    full = [a for a in annual if a["completo"]]
    trend = None
    if len(full) >= 5:
        tot = np.array([a["armonizado"] for a in full])
        if tot.mean() > 0:
            slope = np.polyfit([a["anio"] for a in full], tot, 1)[0]
            trend = {"pct_por_decada": round(float(1000 * slope / tot.mean()), 1),
                     "desde": full[0]["anio"]}

    with_data = [r for r in rows if r["armonizado"] is not None]
    last = with_data[-1] if with_data else rows[-1]
    moy = last["mes_del_anio"] - 1
    last_info = {**last, "mediana_historica": num(st["median"][moy]), "dias": partial_days,
                 "hay_datos": bool(with_data)}
    if same:
        last_info.update(mediana_historica=num(same["mediana"]),
                         mediana_mes_completo=num(st["median"][moy]))
    fit = h["fit"]
    return {
        "area": {"tipo": region.kind, "nombre": region.label, "bbox": region.bbox,
                 "paises": region.countries},
        "estado": dm.status(region),
        "meses": rows, "anual": annual, "tendencia": trend,
        "climatologia": [{"mes_del_anio": m + 1, "nombre": MONTHS[m], "mediana": num(st["median"][m]),
                          "p25": num(st["p25"][m]), "p75": num(st["p75"][m])} for m in range(12)],
        "ultimo_mes": last_info,
        "cobertura": {"modis_hasta": iso(cov["modis"]) if cov["modis"] >= 0 else None,
                      "viirs_hasta": iso(cov["viirs"]) if cov["viirs"] >= 0 else None,
                      "actualizado": cov["actualizado"]},
        "armonizacion": {"a": num(fit["a"]) if np.isfinite(fit["a"]) else None,
                         "b": None if not np.isfinite(fit["b"]) else round(float(fit["b"]), 4),
                         "a_exacto": None if not np.isfinite(fit["a"]) else float(fit["a"]),
                         "sigma": round(float(fit["sigma"]), 4), "origen": fit["origen"],
                         "meses_solapados": fit["meses_solapados"], "validacion": h["cv"]},
        "solape": [{"mes": month_label(int(m)), "modis": round(float(h["modis"][i]), 1),
                    "viirs": int(c["viirs"][i])}
                   for i, m in enumerate(month) if h["from_viirs"][i] and c["have_modis"][i]],
    }


# ------------------------------------------------------------------ catálogo
@app.get("/api/paises")
def paises():
    cat = catalog()
    return {"paises": [{"iso3": k, "nombre": c["nombre"], "bbox": c["bbox"]}
                       for k, c in sorted(cat.countries.items(), key=lambda kv: kv[1]["nombre"])]}


@app.get("/api/paises/{iso3}")
def pais(iso3: str):
    cat = catalog()
    if iso3 not in cat.countries:
        raise HTTPException(404, f"país desconocido: {iso3}")
    provs = cat.by_country.get(iso3, [])
    return {"iso3": iso3, "nombre": cat.countries[iso3]["nombre"], "bbox": cat.countries[iso3]["bbox"],
            "geom": cat.countries[iso3]["geom"],
            "provincias": [{"id": p["id"], "nombre": p["nombre"], "bbox": p["bbox"]} for p in provs]}


@app.get("/api/provincias/{pid}")
def provincia(pid: str):
    cat = catalog()
    if pid not in cat.provinces:
        raise HTTPException(404, f"provincia desconocida: {pid}")
    p = cat.provinces[pid]
    return {"id": pid, "nombre": p["nombre"], "pais": p["pais"], "bbox": p["bbox"], "geom": p["geom"]}


@app.get("/api/ubicar")
def ubicar(lon: float, lat: float):
    hit = catalog().locate(lon, lat)
    if not hit:
        raise HTTPException(404, "ese punto no cae en ningún país")
    return hit


# ------------------------------------------------------------------ datos
@app.get("/api/calendario")
def calendario(pais: str | None = None, provincia: str | None = None, bbox: str | None = None):
    return build_calendar(region_from(pais, provincia, bbox))


@app.get("/api/estado")
def estado(pais: str | None = None, provincia: str | None = None, bbox: str | None = None):
    region = region_from(pais, provincia, bbox)
    for iso3 in region.countries:
        manager().ensure(iso3)
    return manager().status(region)


@app.get("/api/calendario.csv", response_class=PlainTextResponse)
def calendario_csv(pais: str | None = None, provincia: str | None = None, bbox: str | None = None):
    cal = build_calendar(region_from(pais, provincia, bbox))
    cols = ["mes", "armonizado", "bajo", "alto", "origen", "modis_terra", "modis_aqua", "viirs",
            "anomalia", "percentil", "parcial"]
    lines = [",".join(cols)]
    lines += [",".join("" if r[c] is None else str(r[c]) for c in cols) for r in cal["meses"]]
    return PlainTextResponse("\n".join(lines) + "\n", headers={
        "Content-Disposition": 'attachment; filename="calendario_armonizado.csv"'})


@app.get("/api/densidad")
def densidad(inicio: str, fin: str, pais: str | None = None, provincia: str | None = None,
             bbox: str | None = None, px: int = Query(700, ge=50, le=1200)):
    return manager().density(region_from(pais, provincia, bbox), parse_day(inicio), parse_day(fin), px)


@app.get("/api/hotspots.csv")
def hotspots_csv(inicio: str, fin: str, pais: str | None = None, provincia: str | None = None,
                 bbox: str | None = None):
    region = region_from(pais, provincia, bbox)
    rows = manager().csv_rows(region, parse_day(inicio), parse_day(fin))
    return StreamingResponse(rows, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="hotspots_{inicio}_{fin}.csv"'})


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
