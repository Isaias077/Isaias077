from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app as app_module
from server import ingest, store

FIRMS_CSV = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type
-23.10,-58.20,320.1,1.0,1.0,2020-09-01,1405,Terra,MODIS,80,6.03,290.0,25.5,D,0
-23.11,-58.21,310.0,1.0,1.0,2020-09-01,1405,Aqua,MODIS,12,6.03,290.0,9.0,D,0
-23.12,-58.22,300.0,1.0,1.0,2020-09-02,308,Aqua,MODIS,55,6.03,290.0,7.0,N,2
-23.10,-58.20,330.0,0.4,0.4,2020-09-01,1712,N,VIIRS,n,2,295.0,4.5,D,0
-23.10,-58.20,330.0,0.4,0.4,2020-09-01,1712,N,VIIRS,l,2,295.0,1.5,D,0
"""


def test_ingest_parses_both_sensors_into_common_codes():
    cols = ingest.parse(FIRMS_CSV)
    assert cols["sat"] == [0, 1, 1, 2, 2]
    assert cols["conf"] == [2, 0, 1, 1, 0]                  # alta, baja, nominal, nominal, baja
    assert cols["type"] == [0, 0, 2, 0, 0]
    assert cols["night"] == [False, False, True, False, False]
    assert cols["day"][0] == (date(2020, 9, 1) - ingest.EPOCH).days and cols["hhmm"][2] == 308


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Un país sintético: 25 años de fuego estacional dentro de un cuadrado de 2 grados."""
    rng = np.random.default_rng(0)
    cols = {k: [] for k in ("lat", "lon", "day", "hhmm", "sat", "conf", "frp", "night", "type")}

    def add(n, d, sat):
        cols["lat"] += list(rng.uniform(-24, -22, n))
        cols["lon"] += list(rng.uniform(-59, -57, n))
        cols["day"] += [d] * n
        cols["hhmm"] += [1400] * n
        cols["sat"] += [sat] * n
        cols["conf"] += [1] * n
        cols["frp"] += [5.0] * n
        cols["night"] += [False] * n
        cols["type"] += [0] * n

    for year in range(2001, 2026):
        for m in range(1, 13):
            base = int(20 * (1 + 3 * (m in (8, 9))) * rng.uniform(0.7, 1.3))
            d = (date(year, m, 15) - ingest.EPOCH).days
            add(base, d, 0)
            if (year, m) >= (2002, 7):
                add(base, d, 1)
            if year >= 2012:
                add(base * 6, d, 2)
    dtypes = {"lat": "f4", "lon": "f4", "day": "i4", "hhmm": "i2", "sat": "u1", "conf": "u1",
              "frp": "f4", "night": "?", "type": "u1"}
    order = np.argsort(cols["day"], kind="stable")
    np.savez(tmp_path / "Testland.npz", **{k: np.asarray(v, dtype=dtypes[k])[order]
                                           for k, v in cols.items()})
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "DATA_DIR", str(tmp_path))
    app_module.get_store.cache_clear()
    app_module.country_fit.cache_clear()
    return TestClient(app_module.app)


BOX = {"country": "Testland", "bbox": "-59,-24,-57,-22"}


def test_meta_and_static_page(client):
    meta = client.get("/api/meta").json()["paises"][0]
    assert meta["pais"] == "Testland" and meta["desde"] == "2001-01" and meta["hasta"] == "2025-12"
    assert "Calendario de actividad de quema" in client.get("/").text
    assert client.get("/app.js").status_code == 200


def test_calendar_is_continuous_across_the_sensor_change(client):
    cal = client.get("/api/calendar", params=BOX).json()
    assert len(cal["meses"]) == 25 * 12
    by = {m["mes"]: m for m in cal["meses"]}
    assert by["2005-09"]["origen"] == "MODIS convertido" and by["2005-09"]["viirs"] is None
    assert by["2020-09"]["origen"] == "VIIRS" and by["2020-09"]["armonizado"] == by["2020-09"]["viirs"]
    assert by["2005-09"]["bajo"] <= by["2005-09"]["armonizado"] <= by["2005-09"]["alto"]
    pre = np.median([m["armonizado"] for m in cal["meses"] if 2003 <= m["anio"] <= 2011])
    post = np.median([m["armonizado"] for m in cal["meses"] if m["anio"] >= 2013])
    assert 0.8 < pre / post < 1.25                          # VIIRS ve 3 veces más; armonizado, no
    assert cal["armonizacion"]["validacion"]["factor_mensual"] < 1.2
    peak = max(cal["climatologia"], key=lambda c: c["mediana"])
    assert peak["mes_del_anio"] in (8, 9)
    assert len(cal["solape"]) == 14 * 12 - 1 and cal["anual"][0]["completo"]


def test_density_picks_the_sensor_by_era_and_counts_only_the_box(client):
    q = {**BOX, "start": "2020-09-01", "end": "2020-09-30"}
    d = client.get("/api/density", params=q).json()
    assert d["sensor"] == "VIIRS" and d["total"] == sum(d["counts"]) > 0
    assert len(d["counts"]) == d["nx"] * d["ny"]
    old = client.get("/api/density", params={**BOX, "start": "2005-09-01", "end": "2005-09-30"}).json()
    assert old["sensor"] == "MODIS" and 0 < old["total"] < d["total"]
    half = client.get("/api/density", params={**q, "bbox": "-59,-24,-58,-22"}).json()
    assert 0.3 * d["total"] < half["total"] < 0.7 * d["total"]


def test_csv_downloads(client):
    q = {**BOX, "start": "2020-09-01", "end": "2020-09-30"}
    res = client.get("/api/hotspots.csv", params=q)
    lines = res.text.strip().splitlines()
    assert lines[0].startswith("latitude,longitude,acq_date,acq_time,satellite,instrument")
    assert "attachment" in res.headers["content-disposition"]
    assert {ln.split(",")[5] for ln in lines[1:]} == {"MODIS", "VIIRS"}
    assert all(ln.split(",")[2].startswith("2020-09") for ln in lines[1:])
    cal = client.get("/api/calendar.csv", params=BOX).text.strip().splitlines()
    assert cal[0].startswith("mes,armonizado,bajo,alto,origen") and len(cal) == 301


def test_bad_requests_are_rejected_with_a_clear_message(client):
    assert client.get("/api/calendar", params={"country": "Testland", "bbox": "1,2,3"}).status_code == 400
    assert client.get("/api/calendar", params={"country": "Testland", "bbox": "5,0,1,2"}).status_code == 400
    res = client.get("/api/calendar", params={"country": "Nowhere", "bbox": "0,0,1,1"})
    assert res.status_code == 404 and "ingest" in res.json()["detail"]
    assert client.get("/api/density", params={**BOX, "start": "ayer", "end": "2020-01-01"}).status_code == 400
