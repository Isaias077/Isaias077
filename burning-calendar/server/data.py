"""Datos a demanda: cada país se baja de FIRMS la primera vez que se lo necesita.

Por país (data/paises/<ISO3>/):

* puntos/<sensor>_<año>.npz: los hotspots del archivo anual, para descargas CSV.
* cubo/<sensor>_<año>.npz: conteos por celda de 0,05°, día y satélite, solo de fuegos de
  vegetación con confianza nominal o alta. Con esto se arma cualquier área y período.
* reciente.npz: el año en curso, bajado por la API de área y recortado al país.
* estado.json: nombre del país en FIRMS, qué archivos están y hasta cuándo cubre la API.

Los archivos se bajan del más nuevo al más viejo, alternando VIIRS y MODIS, para que
la armonización (que necesita los dos sensores) pueda empezar enseguida.

    python -m server.data PRY ARG     # precarga países sin abrir la app
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np

from . import firms
from .firms import DATASETS, DTYPES, EPOCH, NO_TYPE, SAT_NAMES
from .regions import Region, catalog, cell_ids

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATIC_RES = 0.01
RECENT_REFRESH_H = 6.0
CONF_NAMES = ["low", "nominal", "high"]
_D = np.arange(0, 20000)
_DATES = np.datetime64(EPOCH.isoformat(), "D") + _D
MONTH_OF_DAY = (_DATES.astype("datetime64[M]").astype(int) - 360).astype(np.int32)
DOM_OF_DAY = ((_DATES - _DATES.astype("datetime64[M]")).astype(int) + 1).astype(np.int8)


def month_of(day: int) -> int:
    """Mes (desde enero de 2000) de un día; -1 para días anteriores a 2000."""
    return int(MONTH_OF_DAY[day]) if day >= 0 else -1


def day_number(d: date) -> int:
    return (d - EPOCH).days


def iso(day: int) -> str:
    return (EPOCH + timedelta(days=int(day))).isoformat()


def today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _save_npz(path: str, compressed: bool = False, **arrays) -> None:
    """Escribe a un temporal y lo renombra: un corte a mitad nunca deja un archivo roto."""
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        (np.savez_compressed if compressed else np.savez)(fh, **arrays)
    os.replace(tmp, path)


def _static_keys(a: dict) -> np.ndarray:
    sel = a["type"] != 0
    sel &= a["type"] != NO_TYPE
    ix = np.floor(a["lon"][sel] / STATIC_RES).astype(np.int64)
    iy = np.floor(a["lat"][sel] / STATIC_RES).astype(np.int64)
    return (iy + 20000) * 100000 + (ix + 40000)


def resolve_types(a: dict, static: np.ndarray) -> np.ndarray:
    """Tipo de cada hotspot. Los NRT (sin tipo) cerca de una fuente fija conocida son fijos."""
    t = a["type"].copy()
    unknown = t == NO_TYPE
    if unknown.any():
        ix = np.floor(a["lon"][unknown] / STATIC_RES).astype(np.int64)
        iy = np.floor(a["lat"][unknown] / STATIC_RES).astype(np.int64)
        keys = (iy + 20000) * 100000 + (ix + 40000)
        t[unknown] = np.where(np.isin(keys, static), 2, 0)
    return t


def build_cube(a: dict, types: np.ndarray | None = None) -> dict:
    """Conteos por (celda, día, satélite) de los hotspots que cuentan para el calendario."""
    types = a["type"] if types is None else types
    q = (types == 0) & (a["conf"] >= 1)
    cell = cell_ids(a["lon"][q], a["lat"][q]).astype(np.int64)
    key = (cell << 17) | (a["day"][q].astype(np.int64) << 2) | a["sat"][q].astype(np.int64)
    u, n = np.unique(key, return_counts=True)
    return {"cell": (u >> 17).astype(np.int32), "day": ((u >> 2) & 0x7FFF).astype(np.int16),
            "sat": (u & 3).astype(np.uint8), "n": np.minimum(n, 65535).astype(np.uint16)}


EMPTY_CUBE = {"cell": np.zeros(0, np.int32), "day": np.zeros(0, np.int16),
              "sat": np.zeros(0, np.uint8), "n": np.zeros(0, np.uint16)}


class CountryStore:
    """Los datos de un país en disco, con su estado."""

    def __init__(self, iso3: str, root: str):
        self.iso3 = iso3
        self.dir = os.path.join(root, "paises", iso3)
        os.makedirs(os.path.join(self.dir, "puntos"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "cubo"), exist_ok=True)
        self.lock = threading.RLock()
        self._cube_cache: tuple | None = None
        path = os.path.join(self.dir, "estado.json")
        self.state = {"nombre_firms": None, "archivos": {}, "reciente": None}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self.state.update(json.load(fh))

    def save(self) -> None:
        with self.lock:
            tmp = os.path.join(self.dir, "estado.json.part")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.state, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, os.path.join(self.dir, "estado.json"))

    # -------------------------------------------------------- escritura
    def store_file(self, dataset: str, year: int, text: str | None, status: str) -> None:
        name = f"{dataset}_{year}"
        if text is not None:
            a = firms.parse(text)
            _save_npz(os.path.join(self.dir, "puntos", name + ".npz"), compressed=True, **a)
            _save_npz(os.path.join(self.dir, "cubo", name + ".npz"), static=_static_keys(a),
                      **build_cube(a))
        with self.lock:
            self.state["archivos"][name] = status
            self.save()

    def store_recent(self, a: dict, meta: dict) -> None:
        _save_npz(os.path.join(self.dir, "reciente.npz"), compressed=True, **a)
        with self.lock:
            self.state["reciente"] = meta
            self.save()

    # -------------------------------------------------------- lectura
    def status(self, dataset: str, year: int) -> str | None:
        return self.state["archivos"].get(f"{dataset}_{year}")

    def year_ready(self, dataset: str, year: int) -> bool:
        return self.status(dataset, year) in ("ok", "vacio")

    def static_keys(self) -> np.ndarray:
        parts = []
        for f in sorted(f for f in os.listdir(os.path.join(self.dir, "cubo")) if f.endswith(".npz")):
            with np.load(os.path.join(self.dir, "cubo", f)) as z:
                parts.append(z["static"])
        if not parts:
            return np.zeros(0, np.int64)
        keys, counts = np.unique(np.concatenate(parts), return_counts=True)
        return keys[counts >= 2]

    def cube(self) -> dict:
        """Todos los conteos del país (archivo + año en curso), en memoria."""
        cdir = os.path.join(self.dir, "cubo")
        files = sorted(f for f in os.listdir(cdir) if f.endswith(".npz"))
        recent = os.path.join(self.dir, "reciente.npz")
        version = tuple((f, os.path.getmtime(os.path.join(cdir, f))) for f in files) + (
            os.path.getmtime(recent) if os.path.exists(recent) else 0,)
        if self._cube_cache and self._cube_cache[0] == version:
            return self._cube_cache[1]
        parts = []
        for f in files:
            with np.load(os.path.join(cdir, f)) as z:
                parts.append({k: z[k] for k in EMPTY_CUBE})
        if os.path.exists(recent):
            with np.load(recent) as z:
                a = {k: z[k] for k in DTYPES}
            last_archive = self.archive_last_day()
            keep = a["day"] > last_archive
            a = {k: v[keep] for k, v in a.items()}
            parts.append(build_cube(a, resolve_types(a, self.static_keys())))
        cube = {k: np.concatenate([p[k] for p in parts]) if parts else EMPTY_CUBE[k]
                for k in EMPTY_CUBE}
        self._cube_cache = (version, cube)
        return cube

    def archive_last_day(self) -> int:
        """Fin del año más nuevo que ya está en el archivo anual (los dos sensores).

        No exige que estén todos los años anteriores: se bajan del más nuevo al más viejo,
        y los que faltan se marcan año por año como pendientes.
        """
        for y in range(date.today().year - 1, 1999, -1):
            if all(self.year_ready(ds, y) for ds, first in DATASETS.items() if y >= first):
                return day_number(date(y, 12, 31))
        return -1

    def coverage(self) -> dict:
        arch = self.archive_last_day()
        rec = self.state.get("reciente") or {}
        cov = rec.get("cobertura") or {}
        modis = max(arch, day_number(date.fromisoformat(cov["MODIS"]))) if cov.get("MODIS") else arch
        viirs = max(arch, day_number(date.fromisoformat(cov["VIIRS"]))) if cov.get("VIIRS") else arch
        return {"archivo": arch, "modis": modis, "viirs": viirs,
                "reciente_desde": day_number(date.fromisoformat(rec["inicio"])) if rec.get("inicio") else None,
                "actualizado": rec.get("actualizado")}

    def point_files(self, d0: int, d1: int):
        """Arrays de hotspots que pueden caer entre los días d0 y d1."""
        y0, y1 = (EPOCH + timedelta(days=d0)).year, (EPOCH + timedelta(days=d1)).year
        for dataset in DATASETS:
            for y in range(y0, y1 + 1):
                path = os.path.join(self.dir, "puntos", f"{dataset}_{y}.npz")
                if os.path.exists(path):
                    with np.load(path) as z:
                        yield {k: z[k] for k in DTYPES}
        recent = os.path.join(self.dir, "reciente.npz")
        if os.path.exists(recent):
            with np.load(recent) as z:
                a = {k: z[k] for k in DTYPES}
            keep = a["day"] > self.archive_last_day()
            a = {k: v[keep] for k, v in a.items()}
            a["type"] = resolve_types(a, self.static_keys())
            yield a


class DataManager:
    """Coordina descargas en segundo plano y consultas por área."""

    def __init__(self, root: str = DATA_DIR, workers: int = 4, today=today_utc):
        self.root = root
        self.today = today
        self.cat = catalog()
        self.stores: dict[str, CountryStore] = {}
        self.jobs: dict[str, threading.Thread] = {}
        self.progress: dict[str, dict] = {}
        self.pool = ThreadPoolExecutor(workers)
        self.lock = threading.Lock()

    def store(self, iso3: str) -> CountryStore:
        with self.lock:
            if iso3 not in self.stores:
                self.stores[iso3] = CountryStore(iso3, self.root)
            return self.stores[iso3]

    # -------------------------------------------------------- descargas
    def plan(self, s: CountryStore) -> list[tuple[str, int]]:
        last = self.today().year - 1
        todo = []
        for y in range(last, 1999, -1):
            for dataset, first in DATASETS.items():
                if y >= first and not s.year_ready(dataset, y):
                    todo.append((dataset, y))
        return todo

    def recent_stale(self, s: CountryStore) -> bool:
        rec = s.state.get("reciente")
        if not rec or not rec.get("actualizado"):
            return True
        when = datetime.fromisoformat(rec["actualizado"])
        return (datetime.now(timezone.utc) - when).total_seconds() > RECENT_REFRESH_H * 3600

    def ensure(self, iso3: str) -> None:
        """Arranca la descarga del país si falta algo y no está ya en curso."""
        s = self.store(iso3)
        with self.lock:
            job = self.jobs.get(iso3)
            if job and job.is_alive():
                return
            if s.state.get("nombre_firms") == "" and not self.recent_stale(s):
                return
            if s.state.get("nombre_firms") and not self.plan(s) and not self.recent_stale(s):
                return
            t = threading.Thread(target=self._job, args=(iso3,), daemon=True, name=f"firms-{iso3}")
            self.jobs[iso3] = t
            self.progress[iso3] = {"fase": "iniciando", "archivos_total": 0, "archivos_listos": 0,
                                   "reciente": "pendiente", "error": None}
            t.start()

    def _job(self, iso3: str) -> None:
        s = self.store(iso3)
        p = self.progress[iso3]
        try:
            if s.state.get("nombre_firms") is None:
                p["fase"] = "buscando el país en FIRMS"
                s.state["nombre_firms"] = next(
                    (n for n in self.cat.firms_names(iso3) if firms.name_exists(n)), "")
                s.save()
            todo = self.plan(s) if s.state["nombre_firms"] else []
            p.update(fase="descargando" if todo else "año en curso",
                     archivos_total=len(todo), archivos_listos=0)
            futures = [(y, self.pool.submit(self._file, s, ds, y)) for ds, y in todo]
            last = self.today().year - 1
            recent = None
            for y, f in futures:
                # el año en curso arranca cuando ya se sabe si el archivo del año pasado existe;
                # si no, la API tendría que cubrir también el año pasado
                if recent is None and y < last and self.recent_stale(s):
                    recent = self.pool.submit(self._recent, iso3)
                f.result()
                p["archivos_listos"] += 1
            if recent is None and self.recent_stale(s):
                recent = self.pool.submit(self._recent, iso3)
            if recent:
                p["fase"] = "año en curso"
                recent.result()
            p["fase"] = "listo"
        except Exception as exc:          # se muestra en la app; el resto de los datos sigue sirviendo
            p.update(fase="error", error=str(exc))

    def _file(self, s: CountryStore, dataset: str, year: int) -> None:
        name = s.state["nombre_firms"]
        legacy = os.path.join(self.root, "raw", f"{dataset}_{year}_{name}.csv")
        if os.path.exists(legacy):
            with open(legacy, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        else:
            text = firms.download_archive(dataset, year, name)
        if text is None:
            # el último año puede no estar publicado todavía: se reintenta más adelante
            status = "no_publicado" if year >= self.today().year - 1 else "vacio"
            s.store_file(dataset, year, None, status)
        else:
            s.store_file(dataset, year, text, "ok")
            if os.path.exists(legacy):
                os.remove(legacy)

    def _recent(self, iso3: str) -> None:
        s = self.store(iso3)
        p = self.progress[iso3]
        key = firms.map_key()
        today = self.today()
        start = date(today.year, 1, 1)
        if not all(s.year_ready(ds, today.year - 1) for ds in DATASETS) and s.state["nombre_firms"]:
            start = date(today.year - 1, 1, 1)      # el archivo del año pasado aún no está
        avail = firms.availability(key)
        jobs, coverage = [], {}
        for family in firms.API_SOURCES:
            chunks, covered = firms.plan_chunks(start, today, avail, family)
            jobs += chunks
            coverage[family] = covered.isoformat() if covered else None
        geom = self.cat.country_geom(iso3)
        bbox = tuple(np.round(np.array(geom.bounds) + [-0.05, -0.05, 0.05, 0.05], 2))
        cache = os.path.join(self.root, "raw", "api")
        os.makedirs(cache, exist_ok=True)
        tag = s.state["nombre_firms"] or iso3
        p["reciente"] = f"0/{len(jobs)}"

        def fetch(job):
            source, day0, n = job
            path = os.path.join(cache, f"{source}_{tag}_{day0.isoformat()}_{n}.csv")
            fresh = day0 + timedelta(days=n + 7) > today
            if os.path.exists(path) and not fresh:
                with open(path, encoding="utf-8") as fh:
                    return fh.read()
            text = firms.api_get(firms.area_url_path(key, source, bbox, day0, n))
            if not text.lower().startswith("latitude"):
                raise firms.FirmsError("FIRMS no devolvió datos: " + text.strip()[:60])
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            return text

        parts = []
        with ThreadPoolExecutor(3) as pool:
            for i, text in enumerate(pool.map(fetch, jobs), 1):
                parts.append(firms.parse(text))
                p["reciente"] = f"{i}/{len(jobs)}"
        a = {k: np.concatenate([x[k] for x in parts]) if parts else np.zeros(0, DTYPES[k])
             for k in DTYPES}
        inside = self.cat.clip_mask(iso3)(a["lon"].astype(float), a["lat"].astype(float))
        a = {k: v[inside] for k, v in a.items()}
        order = np.argsort(a["day"], kind="stable")
        s.store_recent({k: v[order] for k, v in a.items()},
                       {"inicio": start.isoformat(), "cobertura": coverage,
                        "actualizado": datetime.now(timezone.utc).isoformat(timespec="minutes")})
        p["reciente"] = "listo"

    # -------------------------------------------------------- consultas
    def status(self, region: Region) -> dict:
        out = []
        for iso3 in region.countries:
            s = self.store(iso3)
            p = dict(self.progress.get(iso3) or {})
            job = self.jobs.get(iso3)
            running = bool(job and job.is_alive())
            out.append({"pais": iso3, "nombre": self.cat.countries[iso3]["nombre"],
                        "en_curso": running, "sin_archivo": s.state.get("nombre_firms") == "",
                        **p})
        return {"paises": out, "en_curso": any(c["en_curso"] for c in out)}

    def coverage(self, region: Region) -> dict:
        covs = [self.store(c).coverage() for c in region.countries]
        return {"modis": min(c["modis"] for c in covs), "viirs": min(c["viirs"] for c in covs),
                "archivo": min(c["archivo"] for c in covs),
                "actualizado": min((c["actualizado"] for c in covs if c["actualizado"]), default=None)}

    def _rows(self, region: Region) -> dict:
        cells = self.cat.region_cells(region)
        parts = []
        for iso3 in region.countries:
            cube = self.store(iso3).cube()
            if cells is not None and cube["cell"].size:
                pos = np.searchsorted(cells, cube["cell"])
                pos[pos >= cells.size] = 0
                keep = cells[pos] == cube["cell"]
                cube = {k: v[keep] for k, v in cube.items()}
            parts.append(cube)
        return {k: np.concatenate([p[k] for p in parts]) for k in EMPTY_CUBE}

    def monthly(self, region: Region, max_dom: int | None = None) -> dict:
        """Conteos mensuales por satélite y qué meses tiene cada sensor.

        Siempre va de noviembre de 2000 al mes actual; los meses que todavía no llegaron
        quedan marcados como sin datos.
        """
        cov = self.coverage(region)
        last_day = max(cov["modis"], cov["viirs"])
        rows = self._rows(region)
        sel = rows["day"].astype(np.int64) <= last_day
        if max_dom is not None:
            sel &= DOM_OF_DAY[rows["day"]] <= max_dom
        first, last = 10, month_of(day_number(self.today()))      # noviembre de 2000
        n = max(last - first + 1, 1)
        idx = MONTH_OF_DAY[rows["day"][sel]] - first
        ok = idx >= 0
        out = {"month": np.arange(first, first + n)}
        for code, name in enumerate(("terra", "aqua", "viirs")):
            m = ok & (rows["sat"][sel] == code)
            out[name] = np.bincount(idx[m], weights=rows["n"][sel][m], minlength=n)[:n].astype(float)
        out.update(self._availability(region, out["month"], cov))
        out["coverage"] = cov
        return out

    def _availability(self, region: Region, month: np.ndarray, cov: dict) -> dict:
        """Meses con datos de cada sensor en todos los países del área."""
        year = 2000 + month // 12
        have = {}
        for dataset, label, last in (("modis", "have_modis", cov["modis"]),
                                     ("viirs-snpp", "have_viirs", cov["viirs"])):
            first_year = DATASETS[dataset]
            ok = np.ones(month.size, dtype=bool)
            for iso3 in region.countries:
                s = self.store(iso3)
                arch = s.coverage()["archivo"]
                ready = np.array([s.year_ready(dataset, int(y)) for y in year])
                in_recent = month > month_of(arch)
                ok &= ready | in_recent
            ok &= year >= first_year
            ok &= month <= month_of(last)
            if dataset == "viirs-snpp":
                ok &= month >= (2012 - 2000) * 12 + 1           # primer mes completo de S-NPP
                # un mes que VIIRS no cubre entero no se usa como VIIRS
                end = EPOCH + timedelta(days=int(last))
                full_last = month_of(last) if (end + timedelta(days=1)).day == 1 \
                    or last >= max(cov["modis"], cov["viirs"]) else month_of(last) - 1
                ok &= month <= full_last
            have[label] = ok
        return have

    def density(self, region: Region, d0: int, d1: int, max_px: int = 700) -> dict:
        cov = self.coverage(region)
        d1 = min(d1, max(cov["modis"], cov["viirs"]))
        use_viirs = d0 >= day_number(date(2012, 1, 20)) and d1 <= cov["viirs"]
        rows = self._rows(region)
        sel = (rows["day"] >= d0) & (rows["day"] <= d1)
        sel &= (rows["sat"] == 2) if use_viirs else (rows["sat"] < 2)
        w, s, e, n = region.bbox
        k = max(1, int(np.ceil((e - w) / 0.05 / max_px)))
        step = 0.05 * k
        nx, ny = max(1, int(np.ceil((e - w) / step))), max(1, int(np.ceil((n - s) / step)))
        cells = rows["cell"][sel].astype(np.int64)
        lon = (cells % 7200 + 0.5) * 0.05 - 180.0
        lat = (cells // 7200 + 0.5) * 0.05 - 90.0
        ix = np.clip(((lon - w) / step).astype(int), 0, nx - 1)
        iy = np.clip(((lat - s) / step).astype(int), 0, ny - 1)
        grid = np.bincount(iy * nx + ix, weights=rows["n"][sel], minlength=nx * ny).astype(int)
        return {"nx": nx, "ny": ny, "bbox": [w, s, w + nx * step, s + ny * step],
                "sensor": "VIIRS" if use_viirs else "MODIS", "total": int(grid.sum()),
                "max": int(grid.max()) if grid.size else 0, "counts": grid.tolist(),
                "viirs_hasta": iso(cov["viirs"])}

    def csv_rows(self, region: Region, d0: int, d1: int):
        import shapely
        geom = None if region.whole_country else region.geom
        if geom is not None:
            shapely.prepare(geom)
        yield "latitude,longitude,acq_date,acq_time,satellite,instrument,confidence,frp,daynight\n"
        for iso3 in region.countries:
            for a in self.store(iso3).point_files(d0, d1):
                m = (a["day"] >= d0) & (a["day"] <= d1) & (a["type"] == 0) & (a["conf"] >= 1)
                if geom is not None and m.any():
                    idx = np.flatnonzero(m)
                    m[idx] = shapely.contains_xy(geom, a["lon"][idx].astype(float),
                                                 a["lat"][idx].astype(float))
                lines = []
                for i in np.flatnonzero(m):
                    sat = int(a["sat"][i])
                    lines.append(f"{a['lat'][i]:.5f},{a['lon'][i]:.5f},{iso(int(a['day'][i]))},"
                                 f"{int(a['hhmm'][i]):04d},{SAT_NAMES[sat]},"
                                 f"{'VIIRS' if sat == 2 else 'MODIS'},{CONF_NAMES[int(a['conf'][i])]},"
                                 f"{a['frp'][i]:.2f},{'N' if a['night'][i] else 'D'}\n")
                    if len(lines) >= 20000:
                        yield "".join(lines)
                        lines = []
                if lines:
                    yield "".join(lines)


if __name__ == "__main__":
    dm = DataManager()
    for code in sys.argv[1:]:
        dm.ensure(code.upper())
    while any(t.is_alive() for t in dm.jobs.values()):
        time.sleep(5)
        for code, p in dm.progress.items():
            print(code, p, flush=True)
