"""Descarga el archivo histórico de hotspots de NASA FIRMS para un país.

    python -m server.ingest --country Paraguay

FIRMS publica sin clave un CSV por país y por año: MODIS (Terra y Aqua) desde
noviembre de 2000 y VIIRS S-NPP (375 m) desde enero de 2012. Este módulo los baja,
los guarda en data/raw/ y los compacta en data/<país>.npz para consultas rápidas.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np

BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"
DATASETS = {"modis": 2000, "viirs-snpp": 2012}       # primer año de cada archivo
EPOCH = date(2000, 1, 1)
SAT_CODES = {"TERRA": 0, "T": 0, "AQUA": 1, "A": 1, "N": 2}
SAT_NAMES = ["Terra", "Aqua", "S-NPP"]
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
            cols["type"].append(int(row.get("type") or 0))
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
    dtypes = {"lat": "f4", "lon": "f4", "day": "i4", "hhmm": "i2", "sat": "u1", "conf": "u1",
              "frp": "f4", "night": "?", "type": "u1"}
    arrays = {k: np.asarray(v, dtype=dtypes[k]) for k, v in merged.items()}
    order = np.argsort(arrays["day"], kind="stable")
    arrays = {k: v[order] for k, v in arrays.items()}
    out = os.path.join(DATA_DIR, f"{name}.npz")
    np.savez_compressed(out, **arrays)
    print(f"{arrays['day'].size:,} hotspots guardados en {out}".replace(",", " "))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--country", required=True, help="nombre en inglés, como en FIRMS")
    ap.add_argument("--last-year", type=int)
    args = ap.parse_args()
    ingest(args.country, args.last_year)
