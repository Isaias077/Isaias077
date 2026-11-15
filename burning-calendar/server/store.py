"""Hotspots de un país en memoria, con consultas por área y por fecha."""
from __future__ import annotations

import calendar
import json
import os
from datetime import date, timedelta

import numpy as np

from .ingest import DATA_DIR, EPOCH, SAT_NAMES

CONF_NAMES = ["low", "nominal", "high"]
VIIRS_FIRST_DAY = (date(2012, 1, 20) - EPOCH).days


def day_number(d: date) -> int:
    return (d - EPOCH).days


def _month_of(day: int) -> int:
    d = EPOCH + timedelta(days=int(day))
    return (d.year - 2000) * 12 + d.month - 1


def _is_month_end(day: int) -> bool:
    d = EPOCH + timedelta(days=int(day))
    return d.day == calendar.monthrange(d.year, d.month)[1]


class Store:
    """Archivo por país más, si existe, el año en curso bajado por la API.

    Guarda hasta qué día cubre cada sensor: VIIRS S-NPP puede cortarse antes que
    MODIS, y un mes que VIIRS no cubre entero se arma con MODIS convertido.
    """

    def __init__(self, path: str):
        with np.load(path) as z:
            self.a = {k: z[k] for k in z.files}
        self.name = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
        # el archivo por país cubre años completos: termina el 31 de diciembre
        last_year = (EPOCH + timedelta(days=int(self.a["day"].max()))).year
        archive_last = day_number(date(last_year, 12, 31))
        self.modis_last_day = self.viirs_last_day = archive_last
        self.updated = None
        recent = path[:-4] + "_reciente.npz"
        if os.path.exists(recent):
            with np.load(recent) as z:
                meta = json.loads(str(z["meta"]))
                new = {k: z[k] for k in self.a}
            keep = new["day"] > archive_last
            self.a = {k: np.concatenate([self.a[k], new[k][keep]]) for k in self.a}
            cov = meta.get("coverage", {})
            if cov.get("MODIS"):
                self.modis_last_day = day_number(date.fromisoformat(cov["MODIS"]))
            if cov.get("VIIRS"):
                self.viirs_last_day = day_number(date.fromisoformat(cov["VIIRS"]))
            self.updated = meta.get("fetched")
        day = self.a["day"]
        dates = np.datetime64(EPOCH.isoformat(), "D") + day
        self.month = dates.astype("datetime64[M]").astype(int) - 360   # meses desde 2000-01
        self.dom = (dates - dates.astype("datetime64[M]")).astype(int) + 1
        self.last_day = max(self.modis_last_day, self.viirs_last_day)
        self.first_month, self.last_month = int(self.month.min()), _month_of(self.last_day)
        # último mes que VIIRS cubre entero (o hasta donde llegan los datos)
        vm = _month_of(self.viirs_last_day)
        full = _is_month_end(self.viirs_last_day) or self.viirs_last_day >= self.last_day
        self.viirs_last_month = vm if full else vm - 1
        self.partial_days = None if _is_month_end(self.last_day) else (EPOCH + timedelta(
            days=self.last_day)).day
        self.bbox = (float(self.a["lon"].min()), float(self.a["lat"].min()),
                     float(self.a["lon"].max()), float(self.a["lat"].max()))

    @property
    def n(self) -> int:
        return int(self.a["day"].size)

    def mask(self, bbox: tuple[float, float, float, float], d0: int | None = None,
             d1: int | None = None, quality: bool = True) -> np.ndarray:
        """Hotspots dentro de bbox = (oeste, sur, este, norte) y entre los días d0 y d1.

        quality deja solo fuegos de vegetación (tipo 0) con confianza nominal o alta:
        fuera quedan volcanes, fuentes fijas como antorchas de gas y detecciones dudosas.
        """
        a = self.a
        m = (a["lon"] >= bbox[0]) & (a["lon"] <= bbox[2]) & (a["lat"] >= bbox[1]) & (a["lat"] <= bbox[3])
        if d0 is not None:
            m &= a["day"] >= d0
        if d1 is not None:
            m &= a["day"] <= d1
        if quality:
            m &= (a["type"] == 0) & (a["conf"] >= 1)
        return m

    def monthly(self, bbox, max_dom: int | None = None) -> dict[str, np.ndarray]:
        """Conteos mensuales por satélite, de first_month a last_month.

        Con max_dom cuenta solo los días 1 a max_dom de cada mes.
        """
        m = self.mask(bbox, d1=self.last_day)
        if max_dom is not None:
            m &= self.dom <= max_dom
        n = self.last_month - self.first_month + 1
        idx = self.month[m] - self.first_month
        sat = self.a["sat"][m]
        out = {name: np.bincount(idx[sat == code], minlength=n).astype(float)
               for code, name in enumerate(("terra", "aqua", "viirs"))}
        out["month"] = np.arange(self.first_month, self.last_month + 1)
        return out

    def density(self, bbox, d0: int, d1: int, nx: int = 160) -> dict:
        """Grilla de conteos para el mapa. Usa VIIRS si todo el rango cae en su era."""
        d1 = min(d1, self.last_day)
        use_viirs = d0 >= VIIRS_FIRST_DAY and d1 <= self.viirs_last_day
        m = self.mask(bbox, d0, d1)
        m &= (self.a["sat"] == 2) if use_viirs else (self.a["sat"] < 2)
        lat_c = np.deg2rad(0.5 * (bbox[1] + bbox[3]))
        ny = int(np.clip(round(nx * (bbox[3] - bbox[1]) / max((bbox[2] - bbox[0]) * np.cos(lat_c), 1e-9)),
                         8, 400))
        ix = np.clip(((self.a["lon"][m] - bbox[0]) / (bbox[2] - bbox[0]) * nx).astype(int), 0, nx - 1)
        iy = np.clip(((self.a["lat"][m] - bbox[1]) / (bbox[3] - bbox[1]) * ny).astype(int), 0, ny - 1)
        grid = np.bincount(iy * nx + ix, minlength=nx * ny)
        return {"nx": nx, "ny": ny, "sensor": "VIIRS" if use_viirs else "MODIS",
                "total": int(m.sum()), "max": int(grid.max()) if grid.size else 0,
                "counts": grid.tolist()}

    def csv_rows(self, bbox, d0: int, d1: int, limit: int = 1_000_000):
        """Líneas CSV de los hotspots del área y el rango, con el sensor de origen."""
        idx = np.flatnonzero(self.mask(bbox, d0, d1))[:limit]
        a = self.a
        yield "latitude,longitude,acq_date,acq_time,satellite,instrument,confidence,frp,daynight\n"
        for start in range(0, idx.size, 20000):
            chunk = idx[start:start + 20000]
            lines = []
            for i in chunk:
                sat = int(a["sat"][i])
                when = EPOCH + timedelta(days=int(a["day"][i]))
                lines.append(f"{a['lat'][i]:.5f},{a['lon'][i]:.5f},{when.isoformat()},"
                             f"{int(a['hhmm'][i]):04d},{SAT_NAMES[sat]},"
                             f"{'VIIRS' if sat == 2 else 'MODIS'},{CONF_NAMES[int(a['conf'][i])]},"
                             f"{a['frp'][i]:.2f},{'N' if a['night'][i] else 'D'}\n")
            yield "".join(lines)


def available() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(DATA_DIR)
                  if f.endswith(".npz") and not f.endswith("_reciente.npz"))
