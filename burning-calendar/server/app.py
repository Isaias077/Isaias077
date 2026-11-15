"""API y servidor web del calendario de actividad de quema.

    uvicorn server.app:app --port 8000
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import harmonize as hz
from .ingest import DATA_DIR, EPOCH
from .store import Store, available, day_number

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

app = FastAPI(title="Calendario de actividad de quema")


@lru_cache(maxsize=4)
def get_store(name: str) -> Store:
    if name not in available():
        raise HTTPException(404, f"No hay datos de '{name}'. Corre: python -m server.ingest "
                                 f"--country {name}")
    return Store(os.path.join(DATA_DIR, f"{name}.npz"))


@lru_cache(maxsize=4)
def country_fit(name: str) -> tuple[float, float, float]:
    """Ajuste MODIS -> VIIRS de todo el país: respaldo para áreas con pocos datos."""
    s = get_store(name)
    c = s.monthly(s.bbox)
    h = hz.harmonize(c["month"], c["terra"], c["aqua"], c["viirs"], s.viirs_last_month)
    return h["fit"]["a"], h["fit"]["b"], h["fit"]["sigma"]


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    try:
        w, s, e, n = (float(v) for v in text.split(","))
    except ValueError:
        raise HTTPException(400, "bbox debe ser oeste,sur,este,norte") from None
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise HTTPException(400, "bbox inválido: se espera oeste < este y sur < norte")
    return w, s, e, n


def parse_day(text: str) -> int:
    try:
        return day_number(date.fromisoformat(text))
    except ValueError:
        raise HTTPException(400, f"fecha inválida: {text}") from None


def month_label(m: int) -> str:
    return f"{2000 + m // 12}-{m % 12 + 1:02d}"


def build_calendar(country: str, bbox) -> dict:
    s = get_store(country)
    c = s.monthly(bbox)
    month = c["month"]
    h = hz.harmonize(month, c["terra"], c["aqua"], c["viirs"], s.viirs_last_month,
                     fallback=country_fit(country))
    st = hz.calendar_stats(month, h["harmonized"])
    rows = [{
        "mes": month_label(int(m)), "anio": 2000 + int(m) // 12, "mes_del_anio": int(m) % 12 + 1,
        "armonizado": round(float(h["harmonized"][i]), 1),
        "bajo": round(float(h["low"][i]), 1), "alto": round(float(h["high"][i]), 1),
        "origen": "VIIRS" if h["from_viirs"][i] else "MODIS convertido",
        "modis_terra": int(c["terra"][i]), "modis_aqua": int(c["aqua"][i]),
        "viirs": int(c["viirs"][i]) if h["from_viirs"][i] else None,
        "anomalia": round(float(st["anomaly"][i]), 3),
        "percentil": round(float(st["rank"][i]), 1),
    } for i, m in enumerate(month)]

    years = sorted({r["anio"] for r in rows})
    complete = [y for y in years if sum(r["anio"] == y for r in rows) == 12]
    annual = [{"anio": y, "armonizado": round(sum(r["armonizado"] for r in rows if r["anio"] == y), 1),
               "completo": y in complete} for y in years]
    tot = np.array([a["armonizado"] for a in annual if a["completo"]])
    yrs = np.array([a["anio"] for a in annual if a["completo"]], dtype=float)
    trend = None
    if tot.size >= 5 and tot.mean() > 0:
        slope = np.polyfit(yrs, tot, 1)[0]
        trend = {"pct_por_decada": round(float(1000 * slope / tot.mean()), 1)}

    last = rows[-1]
    return {
        "pais": country, "bbox": bbox, "total_hotspots": int(sum(r["armonizado"] for r in rows)),
        "meses": rows, "anual": annual, "tendencia": trend,
        "climatologia": [{"mes_del_anio": m + 1, "nombre": MONTHS[m],
                          "mediana": round(float(st["median"][m]), 1),
                          "p25": round(float(st["p25"][m]), 1),
                          "p75": round(float(st["p75"][m]), 1)} for m in range(12)],
        "ultimo_mes": {**last, "mediana_historica": round(float(st["median"][last["mes_del_anio"] - 1]), 1)},
        "armonizacion": {**h["fit"], "validacion": h["cv"],
                         "viirs_desde": month_label(hz.VIIRS_START),
                         "viirs_hasta": month_label(s.viirs_last_month)},
        "solape": [{"mes": month_label(int(m)), "modis": round(float(h["modis"][i]), 1),
                    "viirs": int(c["viirs"][i])}
                   for i, m in enumerate(month) if h["from_viirs"][i]],
    }


@app.get("/api/meta")
def meta():
    out = []
    for name in available():
        s = get_store(name)
        out.append({"pais": name, "hotspots": s.n, "bbox": s.bbox,
                    "desde": month_label(s.first_month), "hasta": month_label(s.last_month),
                    "primer_dia": (EPOCH + timedelta(days=int(s.a["day"][0]))).isoformat(),
                    "ultimo_dia": (EPOCH + timedelta(days=int(s.a["day"][-1]))).isoformat()})
    return {"paises": out}


@app.get("/api/calendar")
def calendar(country: str, bbox: str):
    return build_calendar(country, parse_bbox(bbox))


@app.get("/api/calendar.csv", response_class=PlainTextResponse)
def calendar_csv(country: str, bbox: str):
    cal = build_calendar(country, parse_bbox(bbox))
    cols = ["mes", "armonizado", "bajo", "alto", "origen", "modis_terra", "modis_aqua", "viirs",
            "anomalia", "percentil"]
    lines = [",".join(cols)]
    lines += [",".join("" if r[c] is None else str(r[c]) for c in cols) for r in cal["meses"]]
    return PlainTextResponse("\n".join(lines) + "\n", headers={
        "Content-Disposition": 'attachment; filename="calendario_armonizado.csv"'})


@app.get("/api/density")
def density(country: str, bbox: str, start: str, end: str, nx: int = Query(160, ge=16, le=320)):
    return get_store(country).density(parse_bbox(bbox), parse_day(start), parse_day(end), nx)


@app.get("/api/hotspots.csv")
def hotspots_csv(country: str, bbox: str, start: str, end: str):
    rows = get_store(country).csv_rows(parse_bbox(bbox), parse_day(start), parse_day(end))
    return StreamingResponse(rows, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="hotspots_{start}_{end}.csv"'})


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
