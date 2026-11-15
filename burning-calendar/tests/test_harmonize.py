import numpy as np
import pytest

from server import harmonize as hz


def _series(seed=0, months=312, a=1.6, b=0.94, noise=0.1):
    """Serie sintética desde enero de 2000: VIIRS ve más que MODIS, sobre todo fuegos chicos."""
    rng = np.random.default_rng(seed)
    month = np.arange(months)
    season = 400 * (1 + 4 * np.exp(-0.5 * ((month % 12 - 8) / 1.2) ** 2))
    modis = rng.poisson(season * rng.lognormal(0, 0.3, months)).astype(float)
    terra = np.where(month < hz.AQUA_START, modis * 0.4, modis * 0.4).round()
    aqua = np.where(month < hz.AQUA_START, 0.0, modis - terra)
    viirs = np.expm1(a + b * np.log1p(terra + aqua) + noise * rng.standard_normal(months)).round()
    viirs[month < hz.VIIRS_START] = 0
    return month, terra, aqua, viirs


def test_terra_only_era_is_scaled_to_two_satellites():
    month, terra, aqua, _ = _series()
    modis = hz.terra_to_modis(month, terra, aqua)
    early = month < hz.AQUA_START
    np.testing.assert_allclose(modis[~early], (terra + aqua)[~early])
    np.testing.assert_allclose(modis[early] / terra[early], 2.5, rtol=0.02)   # Terra era 40 %


def test_fit_recovers_the_modis_to_viirs_relation():
    month, terra, aqua, viirs = _series(noise=0.05)
    ov = month >= hz.VIIRS_START
    a, b, sigma = hz.fit_loglog((terra + aqua)[ov], viirs[ov])
    assert a == pytest.approx(1.6, abs=0.15) and b == pytest.approx(0.94, abs=0.03)
    assert sigma == pytest.approx(0.05, abs=0.02)


def test_harmonized_series_uses_viirs_when_it_exists_and_converted_modis_before():
    month, terra, aqua, viirs = _series()
    h = hz.harmonize(month, terra, aqua, viirs, viirs_end=int(month[-1]))
    ov = h["from_viirs"]
    assert ov.sum() == month.size - hz.VIIRS_START
    np.testing.assert_array_equal(h["harmonized"][ov], viirs[ov])
    assert np.all(h["low"][~ov] <= h["harmonized"][~ov]) and np.all(h["harmonized"][~ov] <= h["high"][~ov])
    # la serie convertida queda en la escala de VIIRS: sin salto entre las dos eras
    pre = np.median(h["harmonized"][(month >= 60) & ~ov])
    post = np.median(h["harmonized"][ov])
    assert 0.8 < pre / post < 1.25
    assert np.median(h["modis"][ov]) < 0.5 * post          # sin armonizar, el salto es enorme


def test_cross_validation_shows_the_gain_over_raw_counts():
    month, terra, aqua, viirs = _series()
    cv = hz.harmonize(month, terra, aqua, viirs, int(month[-1]))["cv"]
    assert cv["factor_mensual"] < 1.15 < 2.5 < cv["factor_mensual_sin_armonizar"]
    assert cv["factor_anual"] < cv["factor_mensual"]


def test_sparse_area_falls_back_to_the_country_fit():
    month, terra, aqua, viirs = _series()
    quiet = np.zeros_like(terra)
    quiet[100], quiet[200] = 3, 2                           # casi sin fuego: no se puede ajustar
    h = hz.harmonize(month, quiet, quiet * 0, quiet * 0, int(month[-1]), fallback=(1.6, 0.94, 0.1))
    assert h["fit"]["origen"] == "pais" and h["cv"] is None
    assert h["harmonized"][100] == pytest.approx(np.expm1(1.6 + 0.94 * np.log1p(hz.terra_to_modis(
        month, quiet, quiet * 0)[100])))


def test_anomaly_compares_each_month_with_the_other_years():
    month = np.arange(120)
    value = np.full(120, 100.0)
    value[12 * 5 + 8] = 400.0                               # un septiembre extremo
    st = hz.calendar_stats(month, value)
    assert st["median"][8] == 100.0
    assert st["anomaly"][12 * 5 + 8] == pytest.approx(401 / 101)
    assert st["rank"][12 * 5 + 8] == 100.0
    assert st["anomaly"][12 * 2 + 8] == pytest.approx(1.0)  # el extremo no contamina a los demás
