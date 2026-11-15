"""Acceso a NASA FIRMS: archivos anuales por país (sin clave) y API de área (con clave)."""
from __future__ import annotations

import csv
import io
import os
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

import numpy as np

ARCHIVE = "https://firms.modaps.eosdis.nasa.gov/data/country"
API = "https://firms.modaps.eosdis.nasa.gov/api"
# MAP_KEY del proyecto. La variable de entorno FIRMS_MAP_KEY, si existe, tiene prioridad.
DEFAULT_MAP_KEY = "886fbfe84331ca5604e368739a65e172"
DATASETS = {"viirs-snpp": 2012, "modis": 2000}      # primer año de cada archivo
API_MAX_DAYS = 5
API_SOURCES = {"MODIS": ("MODIS_SP", "MODIS_NRT"), "VIIRS": ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT")}
EPOCH = date(2000, 1, 1)
SAT_CODES = {"TERRA": 0, "T": 0, "AQUA": 1, "A": 1, "N": 2}
SAT_NAMES = ["Terra", "Aqua", "S-NPP"]
NO_TYPE = 255          # las fuentes NRT no traen la columna "type"
DTYPES = {"lat": "f4", "lon": "f4", "day": "i2", "hhmm": "i2", "sat": "u1", "conf": "u1",
          "frp": "f4", "night": "?", "type": "u1"}


class FirmsError(RuntimeError):
    pass


def map_key() -> str:
    return os.environ.get("FIRMS_MAP_KEY") or DEFAULT_MAP_KEY


def _get(url: str, retries: int = 8, ranged: bool = False) -> tuple[int, bytes]:
    """GET con reintentos. Devuelve (código, cuerpo). Los errores nunca muestran la URL."""
    last = "sin respuesta"
    for attempt in range(retries):
        try:
            headers = {"User-Agent": "burning-calendar"}
            if ranged:
                headers["Range"] = "bytes=0-300"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404):
                return exc.code, b""
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            last = type(exc).__name__
        time.sleep(min(2 ** attempt, 30))
    raise FirmsError(f"no se pudo conectar con FIRMS ({last})")


# ------------------------------------------------------------------ archivo por país
def archive_url(dataset: str, year: int, name: str) -> str:
    return f"{ARCHIVE}/{dataset}/{year}/{dataset}_{year}_{name}.csv"


def name_exists(name: str) -> bool:
    """FIRMS tiene archivos con ese nombre de país (prueba unos años con datos casi seguros)."""
    for dataset, year in (("modis", 2020), ("viirs-snpp", 2020), ("modis", 2010)):
        code, body = _get(archive_url(dataset, year, name), retries=6, ranged=True)
        if code in (200, 206) and body.lstrip().lower().startswith(b"latitude"):
            return True
    return False


def download_archive(dataset: str, year: int, name: str) -> str | None:
    """Texto del CSV anual, o None si FIRMS no tiene ese archivo (país sin fuegos ese año)."""
    code, body = _get(archive_url(dataset, year, name))
    if code == 404 or not body.lstrip().lower().startswith(b"latitude"):
        return None
    return body.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ API de área
def api_get(path: str) -> str:
    code, body = _get(f"{API}/{path}")
    if code in (400, 401, 403):
        raise FirmsError(f"FIRMS rechazó el pedido a la API (HTTP {code}); revisá la MAP_KEY")
    return body.decode("utf-8", errors="replace")


def availability(key: str) -> dict[str, tuple[date, date]]:
    text = api_get(f"data_availability/csv/{key}/all")
    if not text.lower().startswith("data_id"):
        raise FirmsError("FIRMS no aceptó la MAP_KEY")
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            out[row["data_id"]] = (date.fromisoformat(row["min_date"]),
                                   date.fromisoformat(row["max_date"]))
        except (KeyError, ValueError):
            continue
    return out


def plan_chunks(start: date, end: date, avail: dict, family: str) -> tuple[list, date | None]:
    """Pedidos de hasta 5 días: SP mientras haya, NRT después.

    Devuelve (pedidos [(fuente, día inicial, días)], último día cubierto por el sensor).
    """
    chunks, covered = [], None
    day = start
    for source in API_SOURCES[family]:
        if source not in avail:
            continue
        lo, hi = avail[source]
        day = max(day, lo)
        stop = min(hi, end)
        while day <= stop:
            n = min(API_MAX_DAYS, (stop - day).days + 1)
            chunks.append((source, day, n))
            day += timedelta(days=n)
        if stop >= start:
            covered = stop if covered is None else max(covered, stop)
    return chunks, covered


def area_url_path(key: str, source: str, bbox: tuple, day0: date, n: int) -> str:
    area = ",".join(f"{v:.2f}" for v in bbox)
    return f"area/csv/{key}/{source}/{area}/{n}/{day0.isoformat()}"


# ------------------------------------------------------------------ lectura
def _confidence(value: str) -> int:
    """Clase de confianza común a los dos sensores: 0 baja, 1 nominal, 2 alta."""
    v = value.strip().lower()
    if v in ("l", "n", "h"):
        return "lnh".index(v)
    try:
        n = int(float(v))
    except ValueError:
        return 1
    return 0 if n < 30 else (1 if n < 80 else 2)


def parse(text: str) -> dict[str, np.ndarray]:
    """CSV de FIRMS (cualquier sensor y fuente) a columnas compactas."""
    cols: dict[str, list] = {k: [] for k in DTYPES}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            sat = SAT_CODES[row["satellite"].strip().upper()]
            y, m, d = (int(v) for v in row["acq_date"].split("-"))
            lat, lon = float(row["latitude"]), float(row["longitude"])
            day = (date(y, m, d) - EPOCH).days
            hhmm, frp = int(row["acq_time"]), float(row["frp"] or 0)
            conf = _confidence(row["confidence"])
        except (KeyError, ValueError, TypeError):
            continue
        t = row.get("type")
        cols["lat"].append(lat)
        cols["lon"].append(lon)
        cols["day"].append(day)
        cols["hhmm"].append(hhmm)
        cols["sat"].append(sat)
        cols["conf"].append(conf)
        cols["frp"].append(frp)
        cols["night"].append(str(row.get("daynight", "D")).strip().upper() == "N")
        cols["type"].append(int(t) if t not in (None, "") else NO_TYPE)
    return {k: np.asarray(v, dtype=DTYPES[k]) for k, v in cols.items()}
