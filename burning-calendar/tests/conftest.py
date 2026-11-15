"""FIRMS simulado: archivos anuales y API con fuego estacional dentro de Uruguay."""
import threading
import time
from datetime import date, timedelta

import numpy as np
import pytest

from server import data as data_module
from server import firms

TODAY = date(2026, 3, 20)
MODIS_HEAD = ("latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,"
              "confidence,version,bright_t31,frp,daynight,type\n")
VIIRS_HEAD = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,"
              "confidence,version,bright_ti5,frp,daynight,type\n")
NRT_HEAD = ("latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,"
            "confidence,version,bright_t31,frp,daynight\n")
AVAIL = {"MODIS_SP": (date(2000, 11, 1), date(2026, 2, 28)), "MODIS_NRT": (date(2026, 3, 1), TODAY),
         "VIIRS_SNPP_SP": (date(2012, 1, 20), date(2026, 2, 14)),
         "VIIRS_SNPP_NRT": (date(2026, 2, 15), date(2026, 3, 10))}
FLARE = (-56.205, -34.805)        # una "antorcha" fija cerca de Montevideo


def _rows(day: date, n: int, sensor: str, rng, nrt=False) -> list[str]:
    out = []
    for _ in range(n):
        lon, lat = rng.uniform(-57.0, -55.5), rng.uniform(-33.3, -32.0)    # interior de Uruguay
        if sensor == "VIIRS":
            row = f"{lat:.4f},{lon:.4f},330,0.4,0.4,{day},1700,N,VIIRS,n,2,290,4.0,D"
        else:
            sat = "Terra" if rng.uniform() < 0.4 or day < date(2002, 7, 1) else "Aqua"
            row = f"{lat:.4f},{lon:.4f},320,1,1,{day},1400,{sat},MODIS,70,6.1,290,9.0,D"
        out.append(row if nrt else row + ",0")
    return out


def fake_year(dataset: str, year: int) -> str | None:
    if year >= TODAY.year:
        return None
    rng = np.random.default_rng(year * 7 + len(dataset))
    sensor = "VIIRS" if dataset.startswith("viirs") else "MODIS"
    rows = []
    for m in range(1, 13):
        if (year, m) < (2000, 11):
            continue
        for dom in (5, 25):
            base = int(10 * (1 + 3 * (m in (8, 9))) * rng.uniform(0.8, 1.2))
            rows += _rows(date(year, m, dom), base * (3 if sensor == "VIIRS" else 1), sensor, rng)
    if sensor == "MODIS":                                   # la antorcha aparece todos los años
        rows += [f"{FLARE[1]},{FLARE[0]},320,1,1,{year}-06-01,1400,Aqua,MODIS,90,6.1,290,40,D,2"] * 3
    return (VIIRS_HEAD if sensor == "VIIRS" else MODIS_HEAD) + "\n".join(rows) + "\n"


def fake_api(path: str) -> str:
    if path.startswith("data_availability"):
        return "data_id,min_date,max_date\n" + "".join(
            f"{k},{a},{b}\n" for k, (a, b) in AVAIL.items())
    source, area, n, day0 = path.split("/")[-4:]
    rng = np.random.default_rng(hash(path) % 2 ** 32)
    nrt = source.endswith("NRT")
    sensor = "VIIRS" if "VIIRS" in source else "MODIS"
    rows = []
    for k in range(int(n)):
        d = date.fromisoformat(day0) + timedelta(days=k)
        if d.day in (5, 25):
            rows += _rows(d, 10 * (3 if sensor == "VIIRS" else 1), sensor, rng, nrt)
        # fuera de Uruguay (Argentina), dentro del rectángulo que se pide: debe descartarse
        rows.append(f"-31.0,-60.0,330,0.4,0.4,{d},1700,N,VIIRS,n,2,290,4.0,D" + ("" if nrt else ",0"))
        if sensor == "MODIS" and d.day == 10:               # NRT junto a la antorcha, sin tipo
            rows.append(f"{FLARE[1]},{FLARE[0]},320,1,1,{d},1400,Aqua,MODIS,90,6.1,290,40,D" +
                        ("" if nrt else ",2"))
    head = NRT_HEAD if nrt else (VIIRS_HEAD if sensor == "VIIRS" else MODIS_HEAD)
    return head + "".join(r + "\n" for r in rows)


@pytest.fixture()
def fake_firms(monkeypatch):
    calls = {"archive": [], "api": [], "names": []}
    lock = threading.Lock()

    def name_exists(name):
        calls["names"].append(name)
        return name == "Uruguay"

    def download_archive(dataset, year, name):
        with lock:
            calls["archive"].append((dataset, year))
        return fake_year(dataset, year)

    def api_get(path):
        with lock:
            calls["api"].append(path)
        return fake_api(path)

    monkeypatch.setattr(firms, "name_exists", name_exists)
    monkeypatch.setattr(firms, "download_archive", download_archive)
    monkeypatch.setattr(firms, "api_get", api_get)
    return calls


@pytest.fixture()
def manager(tmp_path, fake_firms):
    dm = data_module.DataManager(root=str(tmp_path), workers=2, today=lambda: TODAY)
    yield dm


def wait(dm, iso3="URY", timeout=60):
    t0 = time.time()
    while dm.jobs.get(iso3) and dm.jobs[iso3].is_alive():
        assert time.time() - t0 < timeout, "la descarga simulada no terminó"
        time.sleep(0.05)
