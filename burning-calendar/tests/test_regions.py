import numpy as np
import pytest

from server.regions import catalog, cell_centers, cell_ids


def test_catalog_has_countries_and_provinces_in_spanish():
    c = catalog()
    assert len(c.countries) > 240 and len(c.provinces) > 4500
    assert c.countries["BRA"]["nombre"] == "Brasil"
    names = [p["nombre"] for p in c.by_country["PRY"]]
    assert len(names) == 18 and "Boquerón" in names


def test_locate_finds_country_and_province():
    c = catalog()
    asu = c.locate(-57.58, -25.28)
    assert asu["pais"] == "PRY" and c.provinces[asu["provincia"]]["nombre"] == "Asunción"
    assert c.locate(-58.38, -34.60)["pais"] == "ARG"
    assert c.locate(-30.0, -20.0) is None                   # Atlántico


def test_resolve_regions_and_their_cells():
    c = catalog()
    country = c.resolve(country="PRY")
    assert country.whole_country == "PRY" and c.region_cells(country) is None
    rect = c.resolve(bbox=(-58.5, -26.0, -57.0, -24.5))
    assert rect.countries == ["ARG", "PRY"]
    cells = c.region_cells(rect)
    assert cells.size == 30 * 30 and np.all(np.diff(cells) > 0)
    lon, lat = cell_centers(cells)
    assert lon.min() > -58.5 and lon.max() < -57.0 and lat.min() > -26.0 and lat.max() < -24.5
    tiny = c.resolve(province=c.locate(-57.58, -25.28)["provincia"])
    assert 1 <= c.region_cells(tiny).size <= 10            # Asunción: unas pocas celdas
    with pytest.raises(KeyError):
        c.resolve(country="XXX")


def test_cells_roundtrip_and_clip_mask():
    lon, lat = np.array([-57.58, 10.01]), np.array([-25.28, 45.02])
    cx, cy = cell_centers(cell_ids(lon, lat))
    assert np.all(np.abs(cx - lon) <= 0.025) and np.all(np.abs(cy - lat) <= 0.025)
    inside = catalog().clip_mask("PRY")
    # Ciudad del Este, sobre el río, adentro; Clorinda (Argentina), afuera
    assert list(inside(np.array([-54.61, -57.72]), np.array([-25.51, -25.29]))) == [True, False]


def test_firms_name_candidates():
    c = catalog()
    assert c.firms_names("USA")[0] == "United_States"
    assert c.firms_names("PRY") == ["Paraguay"]
