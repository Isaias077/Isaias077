"""Descarga a demanda con FIRMS simulado: orden, recorte, cobertura y consultas."""
import json
import os
from datetime import date

import numpy as np

from server import data as data_module
from server.data import day_number
from server.regions import catalog
from tests.conftest import FLARE, TODAY, wait


def test_country_is_downloaded_on_first_use_newest_years_first(manager, fake_firms):
    manager.ensure("URY")
    wait(manager)
    p = manager.progress["URY"]
    assert p["fase"] == "listo" and p["archivos_listos"] == p["archivos_total"] == 26 + 14
    order = fake_firms["archive"]
    assert order[:2] == [("viirs-snpp", 2025), ("modis", 2025)]     # lo más nuevo primero
    assert order[-1] == ("modis", 2000)
    s = manager.store("URY")
    assert s.state["nombre_firms"] == "Uruguay"
    assert fake_firms["names"][0] == "Uruguay"
    # el año en curso pide solo 2026: el archivo de 2025 ya existía cuando arrancó
    days = [p.split("/")[-1] for p in fake_firms["api"] if p.startswith("area")]
    assert min(days) >= "2026-01-01"
    assert s.state["reciente"]["cobertura"] == {"MODIS": "2026-03-20", "VIIRS": "2026-03-10"}
    # una segunda consulta no vuelve a bajar nada
    n = len(fake_firms["archive"])
    manager.ensure("URY")
    assert not manager.jobs["URY"].is_alive() and len(fake_firms["archive"]) == n


def test_recent_data_is_clipped_to_the_country_and_flares_are_classified(manager):
    manager.ensure("URY")
    wait(manager)
    s = manager.store("URY")
    with np.load(os.path.join(s.dir, "reciente.npz")) as z:
        lon, lat, day, typ = z["lon"], z["lat"], z["day"], z["type"]
    assert not np.isclose(lon, -60.0).any()                         # Argentina, fuera
    flare = np.isclose(lon, FLARE[0], atol=1e-3) & np.isclose(lat, FLARE[1], atol=1e-3)
    nrt = day >= day_number(date(2026, 3, 1))
    assert (flare & nrt).any() and (typ[flare & nrt] == 255).all()  # se guardan sin tipo...
    rows = list(s.point_files(day_number(date(2026, 3, 1)), day_number(TODAY)))
    recent = rows[-1]
    fl = np.isclose(recent["lon"], FLARE[0], atol=1e-3) & np.isclose(recent["lat"], FLARE[1], atol=1e-3)
    assert (recent["type"][fl] == 2).all()                          # ...y se leen como antorcha


def test_coverage_months_and_partial_month(manager):
    manager.ensure("URY")
    wait(manager)
    c = manager.monthly(catalog().resolve(country="URY"))
    month = c["month"]
    assert month[0] == 10 and month[-1] == (2026 - 2000) * 12 + 2  # nov. 2000 a mar. 2026
    assert c["coverage"]["modis"] == day_number(TODAY)
    feb26, mar26 = month.size - 2, month.size - 1
    assert c["have_viirs"][feb26] and not c["have_viirs"][mar26]   # VIIRS llega al 10 de marzo
    assert c["have_modis"][mar26]
    assert not c["have_viirs"][month < (2012 - 2000) * 12 + 1].any()
    assert c["viirs"][feb26] > 2 * (c["terra"][feb26] + c["aqua"][feb26])


def test_pending_years_are_not_counted_as_zero(manager):
    s = manager.store("URY")
    s.state["nombre_firms"] = "Uruguay"
    for ds, y in (("viirs-snpp", 2025), ("modis", 2025), ("viirs-snpp", 2024), ("modis", 2024)):
        from tests.conftest import fake_year
        s.store_file(ds, y, fake_year(ds, y), "ok")
    c = manager.monthly(catalog().resolve(country="URY"))
    y = 2000 + c["month"] // 12
    assert c["have_modis"][y == 2025].all() and c["have_viirs"][y == 2024].all()
    assert not c["have_modis"][y == 2010].any()                     # todavía no descargado
    assert manager.coverage(catalog().resolve(country="URY"))["archivo"] == day_number(date(2025, 12, 31))


def test_province_and_rectangle_are_sums_of_cells(manager):
    manager.ensure("URY")
    wait(manager)
    cat = catalog()
    whole = manager.monthly(cat.resolve(country="URY"))
    provs = cat.by_country["URY"]
    total = sum(manager.monthly(cat.resolve(province=p["id"]))["viirs"].sum() for p in provs)
    # las provincias suman el país, salvo celdas de borde asignadas por su centro
    assert abs(total - whole["viirs"].sum()) < 0.03 * whole["viirs"].sum()
    rect = manager.monthly(cat.resolve(bbox=(-57.5, -33.5, -56.25, -31.0)))
    assert 0.3 * whole["viirs"].sum() < rect["viirs"].sum() < 0.7 * whole["viirs"].sum()


def test_density_and_csv(manager):
    manager.ensure("URY")
    wait(manager)
    region = catalog().resolve(country="URY")
    d = manager.density(region, day_number(date(2025, 8, 1)), day_number(date(2025, 8, 31)))
    assert d["sensor"] == "VIIRS" and d["total"] == sum(d["counts"]) > 0
    d = manager.density(region, day_number(date(2026, 3, 1)), day_number(date(2026, 3, 31)))
    assert d["sensor"] == "MODIS"                                    # VIIRS no llega a fin de mes
    lines = "".join(manager.csv_rows(region, day_number(date(2025, 8, 1)),
                                     day_number(date(2025, 8, 31)))).strip().splitlines()
    assert lines[0].startswith("latitude,longitude,acq_date")
    assert {ln.split(",")[5] for ln in lines[1:]} == {"MODIS", "VIIRS"}
    assert all(ln.split(",")[2].startswith("2025-08") for ln in lines[1:])


def test_country_without_archive_still_gets_the_current_year(manager, fake_firms):
    manager.ensure("PRY")                                   # el FIRMS simulado solo conoce Uruguay
    wait(manager, "PRY")
    s = manager.store("PRY")
    assert s.state["nombre_firms"] == "" and not fake_firms["archive"]
    assert s.state["reciente"]["inicio"] == "2026-01-01"            # solo el año en curso
    st = manager.status(catalog().resolve(country="PRY"))
    assert st["paises"][0]["sin_archivo"]


def test_files_are_written_atomically(tmp_path):
    path = str(tmp_path / "x.npz")
    data_module._save_npz(path, a=np.arange(3))
    assert sorted(os.listdir(tmp_path)) == ["x.npz"]
    with np.load(path) as z:
        assert list(z["a"]) == [0, 1, 2]


def test_state_survives_a_restart(manager, tmp_path):
    manager.ensure("URY")
    wait(manager)
    again = data_module.DataManager(root=str(tmp_path), today=lambda: TODAY)
    s = again.store("URY")
    assert s.state["nombre_firms"] == "Uruguay" and not again.plan(s)
    with open(os.path.join(s.dir, "estado.json"), encoding="utf-8") as fh:
        assert json.load(fh)["archivos"]["modis_2001"] == "ok"
