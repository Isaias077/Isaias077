import pytest
from fastapi.testclient import TestClient

from server import app as app_module
from tests.conftest import wait


@pytest.fixture()
def client(manager, monkeypatch):
    monkeypatch.setattr(app_module, "manager", lambda: manager)
    return TestClient(app_module.app)


def test_catalog_endpoints(client):
    paises = client.get("/api/paises").json()["paises"]
    assert len(paises) > 240 and any(p["nombre"] == "Uruguay" for p in paises)
    ury = client.get("/api/paises/URY").json()
    assert ury["geom"]["type"] in ("Polygon", "MultiPolygon") and len(ury["provincias"]) == 19
    arg = client.get("/api/paises/ARG").json()["provincias"]
    names = [p["nombre"] for p in arg]
    assert len(names) == len(set(names))                     # los dos "Buenos Aires" se distinguen
    pid = ury["provincias"][0]["id"]
    assert client.get(f"/api/provincias/{pid}").json()["pais"] == "URY"
    assert client.get("/api/ubicar", params={"lon": -56.2, "lat": -34.85}).json()["pais"] == "URY"
    assert client.get("/api/ubicar", params={"lon": -30, "lat": -20}).status_code == 404
    assert "Calendario de actividad de quema" in client.get("/").text


def test_calendar_downloads_on_first_use_and_fills_in(client, manager):
    first = client.get("/api/calendario", params={"pais": "URY"}).json()
    assert first["area"]["nombre"] == "Uruguay"
    assert first["estado"]["paises"][0]["pais"] == "URY"
    wait(manager)
    cal = client.get("/api/calendario", params={"pais": "URY"}).json()
    assert not cal["estado"]["en_curso"]
    by = {m["mes"]: m for m in cal["meses"]}
    assert cal["meses"][0]["mes"] == "2000-11" and cal["meses"][-1]["mes"] == "2026-03"
    assert by["2005-08"]["origen"] == "MODIS convertido" and by["2020-08"]["origen"] == "VIIRS"
    assert by["2026-02"]["origen"] == "VIIRS" and by["2026-03"]["origen"] == "MODIS convertido"
    last = cal["ultimo_mes"]
    assert last["mes"] == "2026-03" and last["parcial"] and last["dias"] == 20
    # los otros marzos se cuentan solo hasta el día 20: entra el 5, no el 25
    assert last["mediana_historica"] < 0.7 * last["mediana_mes_completo"]
    assert cal["armonizacion"]["origen"] == "area"
    assert cal["armonizacion"]["validacion"]["factor_mensual"] < 1.3
    assert cal["cobertura"] == {"modis_hasta": "2026-03-20", "viirs_hasta": "2026-03-10",
                                "actualizado": cal["cobertura"]["actualizado"]}
    peak = max(cal["climatologia"], key=lambda c: c["mediana"])
    assert peak["mes_del_anio"] in (8, 9)


def test_pending_months_are_null_while_downloading(client, manager, monkeypatch):
    from server import firms
    import threading
    gate = threading.Event()
    real = firms.download_archive

    def slow(dataset, year, name):
        if year < 2024:
            gate.wait(20)
        return real(dataset, year, name)

    monkeypatch.setattr(firms, "download_archive", slow)
    client.get("/api/estado", params={"pais": "URY"})
    import time
    t0 = time.time()
    while manager.store("URY").status("modis", 2024) is None and time.time() - t0 < 20:
        time.sleep(0.05)
    cal = client.get("/api/calendario", params={"pais": "URY"}).json()
    by = {m["mes"]: m for m in cal["meses"]}
    assert cal["estado"]["en_curso"]
    assert by["2005-08"]["armonizado"] is None and by["2005-08"]["origen"] == "pendiente"
    assert by["2024-08"]["armonizado"] is not None
    assert any(a["armonizado"] is None for a in cal["anual"])
    gate.set()
    wait(manager)


def test_density_csv_and_province(client, manager):
    client.get("/api/estado", params={"pais": "URY"})
    wait(manager)
    q = {"pais": "URY", "inicio": "2025-08-01", "fin": "2025-08-31"}
    d = client.get("/api/densidad", params=q).json()
    assert d["sensor"] == "VIIRS" and d["total"] > 0 and len(d["counts"]) == d["nx"] * d["ny"]
    res = client.get("/api/hotspots.csv", params=q)
    assert "attachment" in res.headers["content-disposition"]
    assert len(res.text.strip().splitlines()) > 100
    pid = client.get("/api/ubicar", params={"lon": -56.5, "lat": -32.6}).json()["provincia"]
    prov = client.get("/api/calendario", params={"provincia": pid}).json()
    assert prov["area"]["tipo"] == "provincia"
    csv = client.get("/api/calendario.csv", params={"provincia": pid}).text.splitlines()
    assert csv[0].startswith("mes,armonizado") and len(csv) == 1 + (2026 - 2000) * 12 - 10 + 3


@pytest.mark.parametrize("params,code,text", [
    ({"bbox": "1,2,3"}, 400, "oeste,sur"),
    ({"bbox": "5,0,1,2"}, 400, "inválido"),
    ({"bbox": "-80,-60,-30,0"}, 400, "demasiado grande"),
    ({"bbox": "-35,-45,-34,-44"}, 400, "mar"),
    ({"pais": "XXX"}, 404, "desconocido"),
    ({}, 404, "falta el área"),
])
def test_bad_requests(client, params, code, text):
    res = client.get("/api/calendario", params=params)
    assert res.status_code == code and text in res.json()["detail"]
