"""Descarga los hotspots de NASA FIRMS para un país.

    python -m server.ingest --country Paraguay            # años completos, sin clave
    python -m server.ingest --country Paraguay --recent   # año en curso, con MAP_KEY

Años completos: FIRMS publica sin clave un CSV por país y por año, MODIS (Terra y
Aqua) desde noviembre de 2000 y VIIRS S-NPP (375 m) desde enero de 2012. Se guardan
en data/raw/ y se compactan en data/<país>.npz.

Año en curso: llega por la API de área de FIRMS, que necesita una MAP_KEY gratuita
(variable de entorno FIRMS_MAP_KEY). Usa el archivo estándar (SP) mientras exista y
el de tiempo casi real (NRT) después, recorta el rectángulo con el contorno del país
y guarda data/<país>_reciente.npz con la cobertura de cada sensor.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np

BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"
DATASETS = {"modis": 2000, "viirs-snpp": 2012}       # primer año de cada archivo
EPOCH = date(2000, 1, 1)
SAT_CODES = {"TERRA": 0, "T": 0, "AQUA": 1, "A": 1, "N": 2}
SAT_NAMES = ["Terra", "Aqua", "S-NPP"]
API = "https://firms.modaps.eosdis.nasa.gov/api"
API_MAX_DAYS = 5
API_SOURCES = {"MODIS": ("MODIS_SP", "MODIS_NRT"), "VIIRS": ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT")}
NO_TYPE = 255          # las fuentes NRT no traen la columna "type"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _download(url: str, path: str, retries: int = 6) -> bool:
    """Baja url a path. Devuelve False si el archivo no existe (404)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "burning-calendar"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = resp.read()
            if not body.lstrip().lower().startswith(b"latitude"):
                return False
            with open(path + ".part", "wb") as fh:
                fh.write(body)
            os.replace(path + ".part", path)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"no se pudo descargar {os.path.basename(path)}")


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


def parse(text: str) -> dict[str, list]:
    cols: dict[str, list] = {k: [] for k in ("lat", "lon", "day", "hhmm", "sat", "conf",
                                             "frp", "night", "type")}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            sat = SAT_CODES[row["satellite"].strip().upper()]
            y, m, d = (int(v) for v in row["acq_date"].split("-"))
            cols["lat"].append(float(row["latitude"]))
            cols["lon"].append(float(row["longitude"]))
            cols["day"].append((date(y, m, d) - EPOCH).days)
            cols["hhmm"].append(int(row["acq_time"]))
            cols["sat"].append(sat)
            cols["conf"].append(_confidence(row["confidence"]))
            cols["frp"].append(float(row["frp"] or 0))
            cols["night"].append(row["daynight"].strip().upper() == "N")
            t = row.get("type")
            cols["type"].append(int(t) if t not in (None, "") else NO_TYPE)
        except (KeyError, ValueError):
            continue
    return cols


def ingest(country: str, last_year: int | None = None, workers: int = 4) -> str:
    last_year = last_year or date.today().year - 1
    raw_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    name = country.replace(" ", "_")
    jobs = [(ds, year) for ds, first in DATASETS.items() for year in range(first, last_year + 1)]

    def fetch(job):
        ds, year = job
        path = os.path.join(raw_dir, f"{ds}_{year}_{name}.csv")
        ok = _download(f"{BASE}/{ds}/{year}/{ds}_{year}_{name}.csv", path)
        print(f"  {ds} {year}: {'ok' if ok else 'sin datos'}", flush=True)
        return path if ok else None

    print(f"Descargando {len(jobs)} archivos de {country}…", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        paths = [p for p in pool.map(fetch, jobs) if p]
    if not paths:
        raise SystemExit(f"FIRMS no tiene archivos para '{country}'. Revisa el nombre en inglés.")

    merged: dict[str, list] = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            part = parse(fh.read())
        for k, v in part.items():
            merged.setdefault(k, []).extend(v)
    arrays = {k: np.asarray(v, dtype=DTYPES[k]) for k, v in merged.items()}
    order = np.argsort(arrays["day"], kind="stable")
    arrays = {k: v[order] for k, v in arrays.items()}
    out = os.path.join(DATA_DIR, f"{name}.npz")
    np.savez_compressed(out, **arrays)
    print(f"{arrays['day'].size:,} hotspots guardados en {out}".replace(",", " "))
    return out


DTYPES = {"lat": "f4", "lon": "f4", "day": "i4", "hhmm": "i2", "sat": "u1", "conf": "u1",
          "frp": "f4", "night": "?", "type": "u1"}


class FirmsKeyError(RuntimeError):
    pass


def _api_get(url: str, retries: int = 8) -> str:
    """GET a la API. Los mensajes de error nunca incluyen la URL, que lleva la clave."""
    last = "sin respuesta"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "burning-calendar"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403):
                raise FirmsKeyError(f"FIRMS rechazó el pedido (HTTP {exc.code})") from None
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            last = type(exc).__name__
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"no se pudo conectar con FIRMS ({last})")


def availability(key: str) -> dict[str, tuple[date, date]]:
    text = _api_get(f"{API}/data_availability/csv/{key}/all")
    if not text.lower().startswith("data_id"):
        raise FirmsKeyError("FIRMS no aceptó la MAP_KEY: " + text.strip()[:80])
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


def static_cells(archive: dict, res: float = 0.01, min_hits: int = 2) -> set:
    """Celdas donde el archivo marcó fuentes que no son fuego de vegetación.

    Sirve para clasificar los datos NRT, que no traen la columna "type": un hotspot
    reciente en el mismo lugar que una antorcha de gas o un volcán se trata igual.
    """
    sel = archive["type"] != 0
    ix = np.floor(archive["lon"][sel] / res).astype(np.int64)
    iy = np.floor(archive["lat"][sel] / res).astype(np.int64)
    keys, counts = np.unique(np.stack([ix, iy], axis=1), axis=0, return_counts=True)
    return {tuple(k) for k in keys[counts >= min_hits]}


def ingest_recent(country: str, key: str | None = None, workers: int = 3,
                  today: date | None = None) -> str:
    """Año en curso por la API, recortado al país y guardado con su cobertura."""
    from .borders import CountryMask

    key = key or os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("Falta la MAP_KEY de FIRMS: definí la variable FIRMS_MAP_KEY. "
                         "Es gratuita: https://firms.modaps.eosdis.nasa.gov/api/map_key/")
    name = country.replace(" ", "_")
    archive_path = os.path.join(DATA_DIR, f"{name}.npz")
    if not os.path.exists(archive_path):
        raise SystemExit(f"Primero bajá los años completos: python -m server.ingest --country {country}")
    with np.load(archive_path) as z:
        archive = {k: z[k] for k in z.files}
    # el archivo por país cubre años completos: el año en curso empieza el 1 de enero
    start = date((EPOCH + timedelta(days=int(archive["day"].max()))).year + 1, 1, 1)
    today = today or datetime.now(timezone.utc).date()
    mask = CountryMask(name)
    area = ",".join(f"{v:.2f}" for v in mask.bbox)

    avail = availability(key)
    jobs, coverage = [], {}
    for family in API_SOURCES:
        chunks, covered = plan_chunks(start, today, avail, family)
        jobs += chunks
        coverage[family] = covered.isoformat() if covered else None
    cache = os.path.join(DATA_DIR, "raw", "api")
    os.makedirs(cache, exist_ok=True)

    def fetch(job):
        source, day0, n = job
        path = os.path.join(cache, f"{source}_{name}_{day0.isoformat()}_{n}.csv")
        recent = day0 + timedelta(days=n + 7) > today       # la última semana se rehace
        if os.path.exists(path) and not recent:
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        text = _api_get(f"{API}/area/csv/{key}/{source}/{area}/{n}/{day0.isoformat()}")
        if not text.lower().startswith("latitude"):
            raise FirmsKeyError("FIRMS no devolvió datos: " + text.strip()[:80])
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return text

    print(f"{len(jobs)} pedidos a la API desde {start} ({', '.join(f'{f} hasta {c}' for f, c in coverage.items())})",
          flush=True)
    merged: dict[str, list] = {k: [] for k in DTYPES}
    with ThreadPoolExecutor(workers) as pool:
        for i, text in enumerate(pool.map(fetch, jobs), 1):
            for k, v in parse(text).items():
                merged[k].extend(v)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    arrays = {k: np.asarray(v, dtype=DTYPES[k]) for k, v in merged.items()}
    keep = mask(arrays["lon"], arrays["lat"])
    arrays = {k: v[keep] for k, v in arrays.items()}
    unknown = arrays["type"] == NO_TYPE
    statics = static_cells(archive)
    cells = zip(np.floor(arrays["lon"][unknown] / 0.01).astype(np.int64),
                np.floor(arrays["lat"][unknown] / 0.01).astype(np.int64))
    arrays["type"][unknown] = [2 if c in statics else 0 for c in cells]
    order = np.argsort(arrays["day"], kind="stable")
    arrays = {k: v[order] for k, v in arrays.items()}
    out = os.path.join(DATA_DIR, f"{name}_reciente.npz")
    meta = {"coverage": coverage, "fetched": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "start": start.isoformat()}
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print(f"{int(keep.sum()):,} hotspots dentro de {country} (de {keep.size:,} en el rectángulo) "
          f"guardados en {out}".replace(",", " "))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--country", required=True, help="nombre en inglés, como en FIRMS")
    ap.add_argument("--last-year", type=int)
    ap.add_argument("--recent", action="store_true",
                    help="bajar el año en curso por la API (necesita FIRMS_MAP_KEY)")
    args = ap.parse_args()
    try:
        if args.recent:
            ingest_recent(args.country)
        else:
            ingest(args.country, args.last_year)
    except FirmsKeyError as exc:
        raise SystemExit(f"Error de FIRMS: {exc}")
