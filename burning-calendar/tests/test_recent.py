"""Año en curso por la API: recorte al país, cobertura por sensor y mes incompleto."""
import json
from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app as app_module
from server import borders, harmonize as hz, ingest, store

AVAIL = """data_id,min_date,max_date
MODIS_NRT,2026-03-01,2026-03-20
MODIS_SP,2000-11-01,2026-02-28
VIIRS_SNPP_NRT,2026-02-15,2026-03-10
VIIRS_SNPP_SP,2012-01-20,2026-02-14
"""
SP_HEAD = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,"
           "confidence,version,bright_ti5,frp,daynight,type\n")
NRT_HEAD = ("latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,"
            "confidence,version,bright_t31,frp,daynight\n")


def test_paraguay_mask_keeps_border_towns_and_drops_neighbours():
    m = borders.CountryMask("Paraguay")
    lon = np.array([-57.58, -54.61, -60.03, -57.72, -58.83, -54.58])
    lat = np.array([-25.28, -25.51, -22.35, -25.29, -27.47, -25.55])
    # Asunción, Ciudad del Este (sobre el río), Filadelfia | Clorinda, Corrientes, Foz do Iguaçu
    assert list(m(lon, lat)) == [True, True, True, False, False, False]
    with pytest.raises(KeyError):
        borders.country_rings("Atlantis")


def test_chunks_use_sp_then_nrt_and_report_coverage():
    avail = {"VIIRS_SNPP_SP": (date(2012, 1, 20), date(2026, 2, 14)),
             "VIIRS_SNPP_NRT": (date(2026, 2, 15), date(2026, 3, 10))}
    chunks, covered = ingest.plan_chunks(date(2026, 1, 1), date(2026, 3, 20), avail, "VIIRS")
    assert chunks[0] == ("VIIRS_SNPP_SP", date(2026, 1, 1), 5)
    assert all(n <= 5 for _, _, n in chunks)
    sp_days = sum(n for s, _, n in chunks if s.endswith("SP"))
    nrt_days = sum(n for s, _, n in chunks if s.endswith("NRT"))
    assert sp_days == 45 and nrt_days == 24                    # 1 ene–14 feb y 15 feb–10 mar
    assert covered == date(2026, 3, 10)                        # VIIRS se corta antes que hoy


def _archive(tmp_path):
    """Archivo sintético de Paraguay, 2001–2025: fuego estacional y una antorcha fija."""
    rng = np.random.default_rng(1)
    cols = {k: [] for k in ingest.DTYPES}

    def add(n, d, sat, lon=None, lat=None, typ=0):
        cols["lon"] += list(rng.uniform(-60.5, -59.5, n) if lon is None else [lon] * n)
        cols["lat"] += list(rng.uniform(-23.0, -22.0, n) if lat is None else [lat] * n)
        cols["day"] += [d] * n
        cols["hhmm"] += [1400] * n
        cols["sat"] += [sat] * n
        cols["conf"] += [1] * n
        cols["frp"] += [5.0] * n
        cols["night"] += [False] * n
        cols["type"] += [typ] * n

    for year in range(2001, 2026):
        for m in range(1, 13):
            for dom in (5, 25):
                d = (date(year, m, dom) - ingest.EPOCH).days
                base = int(15 * rng.uniform(0.8, 1.2))
                add(base, d, 0)
                if (year, m) >= (2002, 7):
                    add(base, d, 1)
                if year >= 2012:
                    add(base * 3, d, 2)
    add(5, (date(2020, 6, 1) - ingest.EPOCH).days, 2, lon=-58.005, lat=-24.005, typ=2)
    arrays = {k: np.asarray(v, dtype=ingest.DTYPES[k]) for k, v in cols.items()}
    order = np.argsort(arrays["day"], kind="stable")
    np.savez(tmp_path / "Paraguay.npz", **{k: v[order] for k, v in arrays.items()})


def _fake_api(url: str) -> str:
    if "/data_availability/" in url:
        return AVAIL
    source, _, n, day0 = url.split("/")[-4:]
    rows = []
    for k in range(int(n)):
        d = date.fromordinal(date.fromisoformat(day0).toordinal() + k).isoformat()
        if "VIIRS" in source:
            for lon, lat in ((-60.0, -22.5), (-59.9, -22.4), (-58.83, -27.47)):  # el último, Argentina
                rows.append(f"{lat},{lon},330,0.4,0.4,{d},1700,N,VIIRS,n,2,290,4.0,D")
        else:
            rows.append(f"-22.5,-60.0,320,1,1,{d},1400,Terra,MODIS,70,6.1,290,9.0,D")
            rows.append(f"-24.005,-58.005,320,1,1,{d},1400,Aqua,MODIS,90,6.1,290,40.0,D")  # antorcha
    head = SP_HEAD if source.endswith("SP") else NRT_HEAD
    tail = ",0" if source.endswith("SP") else ""
    return head + "".join(r + tail + "\n" for r in rows)


@pytest.fixture()
def recent_store(tmp_path, monkeypatch):
    _archive(tmp_path)
    monkeypatch.setattr(ingest, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "DATA_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(ingest, "_api_get", lambda url: calls.append(url) or _fake_api(url))
    ingest.ingest_recent("Paraguay", key="CLAVE-SECRETA", today=date(2026, 3, 20), workers=1)
    app_module.get_store.cache_clear()
    app_module.country_fit.cache_clear()
    return tmp_path, calls


def test_recent_ingest_clips_to_country_classifies_nrt_and_keeps_key_out(recent_store):
    tmp_path, calls = recent_store
    z = np.load(tmp_path / "Paraguay_reciente.npz")
    meta = json.loads(str(z["meta"]))
    assert meta["coverage"] == {"MODIS": "2026-03-20", "VIIRS": "2026-03-10"}
    assert not (np.isclose(z["lon"], -58.83) & np.isclose(z["lat"], -27.47)).any()   # Argentina fuera
    nrt = z["day"] >= (date(2026, 3, 1) - ingest.EPOCH).days     # MODIS NRT: sin columna "type"
    flare = np.isclose(z["lon"], -58.005)
    assert (flare & nrt).any() and (z["type"][flare & nrt] == 2).all()   # junto a una antorcha conocida
    assert (z["type"][~flare] == 0).all()
    assert all("CLAVE-SECRETA" not in p.name for p in (tmp_path / "raw" / "api").iterdir())
    assert len(calls) == 1 + 16 + 14       # disponibilidad + MODIS (59 SP + 20 NRT días) + VIIRS (45 + 24)


def test_store_merges_recent_data_and_tracks_each_sensor(recent_store):
    tmp_path, _ = recent_store
    s = store.Store(str(tmp_path / "Paraguay.npz"))
    assert store.available() == ["Paraguay"]
    assert s.last_month == (2026 - 2000) * 12 + 2                  # marzo de 2026
    assert s.partial_days == 20
    assert s.viirs_last_month == (2026 - 2000) * 12 + 1            # VIIRS cubre febrero entero, no marzo
    d = s.density(s.bbox, store.day_number(date(2026, 3, 1)), store.day_number(date(2026, 3, 31)))
    assert d["sensor"] == "MODIS"                                  # VIIRS no llega al 31
    feb = s.density(s.bbox, store.day_number(date(2026, 2, 1)), store.day_number(date(2026, 2, 28)))
    assert feb["sensor"] == "VIIRS"


def test_partial_month_is_compared_with_the_same_days_of_other_years(recent_store):
    client = TestClient(app_module.app)
    meta = client.get("/api/meta").json()["paises"][0]
    assert meta["ultimo_dia"] == "2026-03-20" and meta["cobertura"]["viirs_hasta"] == "2026-03-10"
    cal = client.get("/api/calendar", params={"country": "Paraguay",
                                              "bbox": "-61,-23.5,-59,-21.5"}).json()
    last = cal["ultimo_mes"]
    assert last["mes"] == "2026-03" and last["parcial"] and last["dias"] == 20
    assert last["origen"] == "MODIS convertido"                    # VIIRS no cubre todo marzo
    # los otros años se cuentan solo hasta el día 20: la mediana es la del día 5, no la de 5 y 25
    assert last["mediana_historica"] < 0.7 * last["mediana_mes_completo"]
    assert sum(m["parcial"] for m in cal["meses"]) == 1
    feb = next(m for m in cal["meses"] if m["mes"] == "2026-02")
    assert feb["origen"] == "VIIRS" and not feb["parcial"]
    clim_mar = cal["climatologia"][2]["mediana"]
    assert clim_mar == pytest.approx(last["mediana_mes_completo"])   # el mes incompleto no la contamina


def test_same_days_comparison():
    month = np.arange(36)
    value = np.where(month % 12 == 2, 10.0, 99.0)
    value[26] = 20.0
    r = hz.same_days_comparison(month, value, 26)
    assert r["mediana"] == 10.0 and r["anomalia"] == pytest.approx(21 / 11)
    assert r["percentil"] == 100.0 and r["anios_de_referencia"] == 2


def test_default_key_is_used_unless_the_environment_overrides_it(tmp_path, monkeypatch):
    _archive(tmp_path)
    monkeypatch.setattr(ingest, "DATA_DIR", str(tmp_path))
    seen = []
    monkeypatch.setattr(ingest, "_api_get", lambda url: seen.append(url) or _fake_api(url))
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    ingest.ingest_recent("Paraguay", today=date(2026, 1, 3), workers=1)
    assert f"/{ingest.DEFAULT_MAP_KEY}/" in seen[0]
    seen.clear()
    monkeypatch.setenv("FIRMS_MAP_KEY", "OTRA")
    ingest.ingest_recent("Paraguay", today=date(2026, 1, 3), workers=1)
    assert "/OTRA/" in seen[0] and all(ingest.DEFAULT_MAP_KEY not in u for u in seen)
