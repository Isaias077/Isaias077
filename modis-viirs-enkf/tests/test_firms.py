import os
from datetime import date

import numpy as np
import pytest

from frp_enkf import firms
from frp_enkf.sensors import MODIS, VIIRS, pixel_footprints

VIIRS_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_ti5,frp,daynight
-19.50,144.60,340.1,0.40,0.38,2026-11-08,0412,N20,nominal,2.0NRT,295.0,12.5,D
-19.51,144.61,345.0,0.40,0.38,2026-11-08,0412,N20,high,2.0NRT,296.0,30.0,D
-19.52,144.62,330.0,0.40,0.38,2026-11-08,0413,N20,low,2.0NRT,294.0,2.0,D
-19.50,144.60,320.0,0.45,0.40,2026-11-08,1554,N20,nominal,2.0NRT,285.0,6.0,N
-19.50,144.60,320.0,0.50,0.45,2026-11-08,0503,N21,nominal,2.0NRT,285.0,9.0,D
-10.00,100.00,320.0,0.50,0.45,2026-11-08,0503,N21,nominal,2.0NRT,285.0,9.0,D
"""
MODIS_CSV = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,confidence,version,bright_t31,frp,daynight
-19.50,144.60,330.0,2.1,1.4,2026-11-08,36,T,80,6.1NRT,300.0,95.0,D
-19.50,144.63,315.0,2.1,1.4,2026-11-08,36,T,12,6.1NRT,300.0,20.0,D
"""
BBOX = (144.35, -19.75, 144.85, -19.25)


def _domain():
    return firms.Domain(*BBOX, t0=np.datetime64("2026-11-08"))


def test_parse_filters_bbox_and_low_confidence():
    d = firms.parse_csv(VIIRS_CSV, "VIIRS_NOAA20_NRT", BBOX)
    assert len(d) == 4 and d.sensor is VIIRS
    assert set(d.platform) == {"NOAA-20", "NOAA-21"}
    assert d.time[0] == np.datetime64("2026-11-08T04:12")
    assert list(d.day) == [True, True, False, True]
    m = firms.parse_csv(MODIS_CSV, "MODIS_NRT", BBOX)
    assert len(m) == 1 and m.sensor is MODIS and m.platform[0] == "Terra"
    assert m.time[0] == np.datetime64("2026-11-08T00:36")     # "36" es 00:36
    assert len(firms.parse_csv(MODIS_CSV, "MODIS_NRT", BBOX, drop_low_confidence=False)) == 2


def test_passes_are_split_by_platform_and_time():
    d = firms.parse_csv(VIIRS_CSV, "VIIRS_NOAA20_NRT", BBOX)
    passes = firms.split_passes(d)
    assert [(str(p.platform[0]), len(p)) for p in passes] == [
        ("NOAA-20", 2), ("NOAA-21", 1), ("NOAA-20", 1)]


def test_domain_projection_and_solar_time():
    dom = _domain()
    ny, nx = dom.shape
    assert 140 <= nx <= 142 and 147 <= ny <= 149           # ~52.5 km x ~55.3 km
    x, y = dom.to_km(np.array([BBOX[1]]), np.array([BBOX[0]]))
    assert x[0] == pytest.approx(0.0) and y[0] == pytest.approx(0.0)
    h = dom.hours(np.array(["2026-11-08T04:00"], dtype="datetime64[m]"))[0]
    assert h == pytest.approx(4.0 + 144.6 / 15.0)           # ~13:38 solar
    assert dom.utc(h) == np.datetime64("2026-11-08T04:00")


def test_scan_axis_is_tilted_a_few_degrees_and_flips_with_node():
    asc = firms.scan_axis_deg(-19.5, 98.7, ascending=True)
    desc = firms.scan_axis_deg(-19.5, 98.7, ascending=False)
    assert 7.0 < asc < 11.0 and desc == pytest.approx(-asc)


def test_pixel_footprint_sees_a_fire_at_its_centre():
    shape = (40, 40)
    x = np.zeros(shape)
    x[20, 20] = 50.0                                         # celda centrada en 7.6875 km
    for sensor, ds, dt in ((VIIRS, 0.45, 0.40), (MODIS, 2.0, 1.4)):
        H = pixel_footprints(np.array([7.6875]), np.array([7.6875]), np.array([ds]),
                             np.array([dt]), 9.0, shape, sensor.triangular_psf, nsub=8)
        seen = float((H @ x.ravel())[0])
        assert 0.8 * 50.0 < seen <= 50.0
    # fuera de la grilla: fila vacía, sin error
    H = pixel_footprints(np.array([-30.0]), np.array([5.0]), np.array([0.4]), np.array([0.4]),
                         9.0, shape, False)
    assert H.shape == (1, 1600) and H.nnz == 0


def test_observation_has_real_footprints_and_inferred_non_detections():
    dom = _domain()
    d = firms.parse_csv(VIIRS_CSV, "VIIRS_NOAA20_NRT", BBOX)
    obs = firms.passes_from_detections([d], dom)
    assert [o.platform for o in obs] == ["NOAA-20", "NOAA-21", "NOAA-20"]
    o = obs[0]
    assert o.detected.sum() == 2 and np.nansum(o.frp) == pytest.approx(42.5)
    assert o.thr_mw == pytest.approx(VIIRS.thr_density * 0.40 * 0.38)
    assert (~o.detected).sum() > 1000                        # malla de no-detecciones
    # ningún píxel "sin detección" se superpone con los detectados
    burning = np.zeros(dom.shape[0] * dom.shape[1])
    burning[o.H[o.detected].indices] = 1.0
    overlap = (o.H[~o.detected] @ burning) / np.asarray(o.H[~o.detected].sum(axis=1)).ravel()
    assert overlap.max() <= 0.5
    only_det = firms.passes_from_detections([d], dom, non_detections=False)
    assert only_det[0].detected.all()


def test_fetch_area_needs_a_key_and_never_leaks_it(monkeypatch, tmp_path):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(firms.FirmsError, match="MAP_KEY"):
        firms.fetch_area("VIIRS_NOAA20_NRT", BBOX, date(2026, 11, 8), 3)

    calls = []
    monkeypatch.setattr(firms, "_get", lambda url: calls.append(url) or VIIRS_CSV)
    text = firms.fetch_area("VIIRS_NOAA20_NRT", BBOX, date(2020, 1, 1), 7, "SECRETA", str(tmp_path))
    assert len(calls) == 2                                   # 7 días = pedidos de 5 + 2
    assert calls[0].endswith("/VIIRS_NOAA20_NRT/144.35,-19.75,144.85,-19.25/5/2020-01-01")
    assert calls[1].endswith("/2/2020-01-06")
    assert text.count("latitude") == 1 and len(text.strip().splitlines()) == 13
    assert all("SECRETA" not in name for name in os.listdir(tmp_path))
    firms.fetch_area("VIIRS_NOAA20_NRT", BBOX, date(2020, 1, 1), 7, "SECRETA", str(tmp_path))
    assert len(calls) == 2                                   # segunda vez: sale de la caché

    monkeypatch.setattr(firms, "_get", lambda url: "Invalid MAP_KEY.")
    with pytest.raises(firms.FirmsError) as err:
        firms.fetch_area("MODIS_NRT", BBOX, date(2020, 1, 1), 1, "SECRETA")
    assert "SECRETA" not in str(err.value)


def test_fetch_recent_builds_keyless_url(monkeypatch):
    calls = []
    monkeypatch.setattr(firms, "_get", lambda url: calls.append(url) or MODIS_CSV)
    firms.fetch_recent("MODIS_NRT", "7d")
    assert calls == [firms.BASE_URL + "/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv"]
    with pytest.raises(firms.FirmsError):
        firms.fetch_recent("MODIS_SP", "7d")
