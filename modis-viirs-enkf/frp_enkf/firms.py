"""Datos reales de NASA FIRMS: descarga, lectura y conversión a pasadas del filtro.

Hay dos formas de bajar datos:

* `fetch_area`: la API de área de FIRMS. Sirve para cualquier fecha y necesita una
  MAP_KEY gratuita (https://firms.modaps.eosdis.nasa.gov/api/map_key/). La clave se
  lee del argumento o de la variable de entorno FIRMS_MAP_KEY.
* `fetch_recent`: los archivos globales de las últimas 24 h, 48 h o 7 días. No
  necesitan clave.

FIRMS solo lista detecciones. Lo que el filtro necesita además es saber dónde el
sensor miró y no vio fuego; `passes_from_detections` lo reconstruye con una malla
de píxeles "sin detección" para cada pasada que tuvo al menos una detección en el
dominio. Es una aproximación: no hay máscara de nubes ni límites del barrido.
"""
from __future__ import annotations

import csv
import io
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
from scipy import sparse

from .model import CELL_KM
from .sensors import MODIS, VIIRS, Observation, Sensor, lattice_footprints, pixel_footprints

BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
API_MAX_DAYS = 5      # rango máximo que la API de área acepta por pedido

# fuente de FIRMS -> (sensor, carpeta y prefijo de los archivos globales sin clave)
SOURCES: dict[str, tuple[Sensor, str | None, str | None]] = {
    "MODIS_NRT": (MODIS, "modis-c6.1", "MODIS_C6_1"),
    "MODIS_SP": (MODIS, None, None),
    "VIIRS_SNPP_NRT": (VIIRS, "suomi-npp-viirs-c2", "SUOMI_VIIRS_C2"),
    "VIIRS_SNPP_SP": (VIIRS, None, None),
    "VIIRS_NOAA20_NRT": (VIIRS, "noaa-20-viirs-c2", "J1_VIIRS_C2"),
    "VIIRS_NOAA20_SP": (VIIRS, None, None),
    "VIIRS_NOAA21_NRT": (VIIRS, "noaa-21-viirs-c2", "J2_VIIRS_C2"),
}
DEFAULT_SOURCES = ("MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")

# código de la columna "satellite" -> (plataforma, inclinación orbital, pasada diurna ascendente)
PLATFORMS = {
    "T": ("Terra", 98.2, False), "TERRA": ("Terra", 98.2, False),
    "A": ("Aqua", 98.2, True), "AQUA": ("Aqua", 98.2, True),
    "N": ("S-NPP", 98.7, True), "N20": ("NOAA-20", 98.7, True), "1": ("NOAA-20", 98.7, True),
    "N21": ("NOAA-21", 98.7, True), "2": ("NOAA-21", 98.7, True),
}


class FirmsError(RuntimeError):
    pass


@dataclass
class Detections:
    """Detecciones de un sensor. Todos los arreglos tienen el mismo largo."""

    sensor: Sensor
    lat: np.ndarray
    lon: np.ndarray
    frp: np.ndarray            # MW
    scan: np.ndarray           # tamaño del píxel a lo largo del barrido [km]
    track: np.ndarray          # tamaño del píxel a lo largo de la trayectoria [km]
    time: np.ndarray           # datetime64[m], UTC
    platform: np.ndarray       # "Terra", "Aqua", "S-NPP", "NOAA-20", "NOAA-21"
    day: np.ndarray            # bool: pasada diurna

    def __len__(self) -> int:
        return self.lat.size

    def subset(self, mask: np.ndarray) -> "Detections":
        return Detections(self.sensor, *(getattr(self, f)[mask] for f in (
            "lat", "lon", "frp", "scan", "track", "time", "platform", "day")))


# ----------------------------------------------------------------------- descarga
def _get(url: str, retries: int = 4) -> str:
    """GET con reintentos. Los errores nunca incluyen la URL, que puede llevar la clave."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "frp-enkf"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404):
                raise FirmsError(f"FIRMS respondió HTTP {exc.code}") from None
            last = exc
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(2 ** attempt)
    raise FirmsError(f"no se pudo conectar con FIRMS: {type(last).__name__}") from None


def _check_csv(text: str) -> str:
    if not text.lstrip().lower().startswith("latitude"):
        raise FirmsError("FIRMS no devolvió un CSV: " + text.strip()[:120])
    return text


def _cached(path: str | None, max_age_h: float, download) -> str:
    if path and os.path.exists(path) and time.time() - os.path.getmtime(path) < 3600 * max_age_h:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    text = download()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def fetch_area(source: str, bbox: tuple[float, float, float, float], start: date, days: int,
               map_key: str | None = None, cache_dir: str | None = None) -> str:
    """CSV de la API de área para bbox = (oeste, sur, este, norte) y `days` días desde `start`."""
    if source not in SOURCES:
        raise FirmsError(f"fuente desconocida: {source}. Opciones: {', '.join(SOURCES)}")
    map_key = map_key or os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        raise FirmsError("falta la MAP_KEY de FIRMS: pásala con --map-key o en FIRMS_MAP_KEY. "
                         "Es gratuita: https://firms.modaps.eosdis.nasa.gov/api/map_key/")
    area = ",".join(f"{v:g}" for v in bbox)
    header, lines = None, []
    for off in range(0, days, API_MAX_DAYS):
        n = min(API_MAX_DAYS, days - off)
        day0 = start + timedelta(days=off)
        url = f"{BASE_URL}/api/area/csv/{map_key}/{source}/{area}/{n}/{day0.isoformat()}"
        path = cache_dir and os.path.join(
            cache_dir, f"{source}_{area.replace(',', '_')}_{day0.isoformat()}_{n}d.csv")
        # los días pasados no cambian; el día en curso sí
        age = 1.0 if day0 + timedelta(days=n) > datetime.now(timezone.utc).date() else 1e9
        chunk = _cached(path, age, lambda: _check_csv(_get(url))).strip().splitlines()
        header = header or chunk[0]
        lines += chunk[1:]
    return "\n".join([header] + lines) + "\n"


def fetch_recent(source: str, period: str = "7d", cache_dir: str | None = None) -> str:
    """CSV global de las últimas 24 h, 48 h o 7 días. No necesita clave."""
    if period not in ("24h", "48h", "7d"):
        raise FirmsError("period debe ser 24h, 48h o 7d")
    _, folder, prefix = SOURCES.get(source, (None, None, None))
    if folder is None:
        raise FirmsError(f"{source} no tiene archivo global reciente; usa la API con MAP_KEY")
    url = f"{BASE_URL}/data/active_fire/{folder}/csv/{prefix}_Global_{period}.csv"
    path = cache_dir and os.path.join(cache_dir, f"{prefix}_Global_{period}.csv")
    return _cached(path, 3.0, lambda: _check_csv(_get(url)))


# ----------------------------------------------------------------------- lectura
def parse_csv(text: str, source: str, bbox: tuple[float, float, float, float] | None = None,
              drop_low_confidence: bool = True) -> Detections:
    """Convierte un CSV de FIRMS en Detections, opcionalmente recortado a bbox."""
    sensor = SOURCES[source][0]
    cols: dict[str, list] = {k: [] for k in ("lat", "lon", "frp", "scan", "track", "time",
                                             "platform", "day")}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            lat, lon, frp = float(row["latitude"]), float(row["longitude"]), float(row["frp"])
            scan, track = float(row["scan"]), float(row["track"])
        except (KeyError, ValueError, TypeError):
            continue
        if bbox and not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
            continue
        conf = str(row.get("confidence", "")).strip().lower()
        if drop_low_confidence and (conf in ("l", "low") or (conf.isdigit() and int(conf) < 30)):
            continue
        if not frp > 0:
            continue
        sat = str(row.get("satellite", "")).strip().upper()
        if sat not in PLATFORMS:
            continue
        hhmm = str(row["acq_time"]).strip().replace(":", "").zfill(4)
        when = np.datetime64(f"{row['acq_date'].strip()}T{hhmm[:2]}:{hhmm[2:]}", "m")
        for k, v in zip(cols, (lat, lon, frp, scan, track, when, PLATFORMS[sat][0],
                               str(row.get("daynight", "D")).strip().upper() != "N")):
            cols[k].append(v)
    return Detections(sensor, np.array(cols["lat"], float), np.array(cols["lon"], float),
                      np.array(cols["frp"], float), np.array(cols["scan"], float),
                      np.array(cols["track"], float),
                      np.array(cols["time"], dtype="datetime64[m]"),
                      np.array(cols["platform"], dtype=object), np.array(cols["day"], bool))


# ----------------------------------------------------------------------- geometría
@dataclass
class Domain:
    """Grilla de 375 m sobre un rectángulo geográfico, con proyección local plana."""

    west: float
    south: float
    east: float
    north: float
    t0: np.datetime64          # medianoche UTC del primer día

    @property
    def _km_per_deg(self) -> tuple[float, float]:
        lat_c = np.deg2rad(0.5 * (self.south + self.north))
        return 111.32 * float(np.cos(lat_c)), 110.57

    @property
    def shape(self) -> tuple[int, int]:
        kx, ky = self._km_per_deg
        return (int(np.ceil((self.north - self.south) * ky / CELL_KM)),
                int(np.ceil((self.east - self.west) * kx / CELL_KM)))

    @property
    def solar_offset_h(self) -> float:
        """Horas a sumar a la hora UTC para obtener la hora solar local."""
        return 0.5 * (self.west + self.east) / 15.0

    def to_km(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kx, ky = self._km_per_deg
        return (np.asarray(lon) - self.west) * kx, (np.asarray(lat) - self.south) * ky

    def hours(self, when: np.ndarray) -> np.ndarray:
        """Horas solares locales desde la medianoche UTC del primer día."""
        minutes = (np.asarray(when, dtype="datetime64[m]") - self.t0.astype("datetime64[m]"))
        return minutes.astype(float) / 60.0 + self.solar_offset_h

    def utc(self, hours: float) -> np.datetime64:
        return self.t0.astype("datetime64[m]") + np.timedelta64(
            int(round((hours - self.solar_offset_h) * 60)), "m")


def scan_axis_deg(lat_deg: float, inclination_deg: float, ascending: bool) -> float:
    """Ángulo del eje de barrido respecto del este (antihorario), para una órbita polar.

    El rumbo de la trayectoria sale de sin(az) = cos(i) / cos(lat); el barrido es
    perpendicular. Se ignora la corrección por rotación terrestre (unos pocos grados).
    """
    s = np.cos(np.deg2rad(inclination_deg)) / np.cos(np.deg2rad(lat_deg))
    az = float(np.rad2deg(np.arcsin(np.clip(s, -1.0, 1.0))))
    return -az if ascending else az


def split_passes(det: Detections, gap_min: float = 20.0) -> list[Detections]:
    """Separa las detecciones en pasadas: misma plataforma y huecos menores a gap_min."""
    out = []
    for platform in sorted(set(det.platform)):
        d = det.subset(det.platform == platform)
        order = np.argsort(d.time, kind="stable")
        d = d.subset(order)
        gaps = np.diff(d.time).astype("timedelta64[m]").astype(float)
        for idx in np.split(np.arange(len(d)), np.flatnonzero(gaps > gap_min) + 1):
            out.append(d.subset(idx))
    return sorted(out, key=lambda p: p.time[0])


def pass_to_observation(p: Detections, dom: Domain, non_detections: bool = True) -> Observation:
    """Convierte una pasada de FIRMS en una Observation con su H por huella real."""
    platform = str(p.platform[0])
    _, incl, day_asc = next(v for v in PLATFORMS.values() if v[0] == platform)
    lat_c = 0.5 * (dom.south + dom.north)
    heading = scan_axis_deg(lat_c, incl, ascending=(bool(p.day[0]) == day_asc))
    cx, cy = dom.to_km(p.lat, p.lon)
    tri = p.sensor.triangular_psf
    H = pixel_footprints(cx, cy, p.scan, p.track, heading, dom.shape, tri)
    seen = np.asarray(H.sum(axis=1)).ravel() > 0      # la huella toca el dominio
    H, cx, cy, frp = H[seen], cx[seen], cy[seen], p.frp[seen]
    ds, dt = float(np.median(p.scan)), float(np.median(p.track))
    centers = np.column_stack([cx, cy])
    detected = np.ones(frp.size, dtype=bool)

    if non_detections and frp.size:
        # malla de píxeles del mismo tamaño que la pasada; se descartan los que pisan
        # una detección, porque ahí el sensor sí vio fuego
        Hl, cl = lattice_footprints(ds, dt, heading, dom.shape, tri)
        phi = np.deg2rad(heading)
        du = ((cl[:, None, 0] - cx[None]) * np.cos(phi) + (cl[:, None, 1] - cy[None]) * np.sin(phi)) / ds
        dv = (-(cl[:, None, 0] - cx[None]) * np.sin(phi) + (cl[:, None, 1] - cy[None]) * np.cos(phi)) / dt
        free = ~((np.abs(du) < (1.5 if tri else 1.0)) & (np.abs(dv) < 1.0)).any(axis=1)
        H = sparse.vstack([H, Hl[free]]).tocsr()
        centers = np.vstack([centers, cl[free]])
        detected = np.concatenate([detected, np.zeros(free.sum(), dtype=bool)])
        frp = np.concatenate([frp, np.full(free.sum(), np.nan)])

    when = p.time[0] + (p.time[-1] - p.time[0]) // 2
    return Observation(p.sensor, platform, float(dom.hours(when)), float("nan"), H, centers,
                       ds, dt, p.sensor.thr_density * ds * dt,
                       np.ones(detected.size, dtype=bool), detected, frp)


def passes_from_detections(dets: list[Detections], dom: Domain,
                           non_detections: bool = True) -> list[Observation]:
    """Todas las pasadas de todos los sensores, ordenadas en el tiempo."""
    obs = [pass_to_observation(p, dom, non_detections)
           for d in dets if len(d) for p in split_passes(d)]
    return sorted((o for o in obs if o.detected.any()), key=lambda o: o.time_h)
