import numpy as np
import pytest

from frp_enkf.letkf import gaspari_cohn, letkf_update
from frp_enkf.model import FireParams, diurnal, step
from frp_enkf.sensors import MODIS, VIIRS, build_footprints, grid_naive, observe


def test_pixel_growth_matches_published_geometry():
    assert MODIS.pixel_size(0.0) == pytest.approx((1.0, 1.0), abs=1e-6)
    ds, dt = MODIS.pixel_size(55.0)
    assert ds == pytest.approx(4.83, abs=0.05) and dt == pytest.approx(2.01, abs=0.05)
    assert VIIRS.pixel_size(0.0) == pytest.approx((0.375, 0.375), abs=1e-6)
    # la agregación mantiene el píxel de VIIRS por debajo de ~0.8 km en todo el barrido
    assert max(max(VIIRS.pixel_size(t)) for t in np.linspace(0, 56, 113)) < 0.85


@pytest.mark.parametrize("sensor,theta", [(MODIS, 0.0), (MODIS, 50.0), (VIIRS, 0.0), (VIIRS, 40.0)])
def test_observation_operator_conserves_energy(sensor, theta):
    H, centers, ds, dt = build_footprints(sensor, theta, 10.0, (40, 40), offset=(0.3, 0.7))
    np.testing.assert_allclose(np.asarray(H.sum(axis=0)).ravel(), 1.0, atol=1e-9)
    x = np.random.default_rng(0).gamma(1.0, 5.0, 1600)
    assert (H @ x).sum() == pytest.approx(x.sum())
    assert centers.shape == (H.shape[0], 2)


def test_modis_misses_small_fires_at_scan_edge():
    rng = np.random.default_rng(1)
    x = np.zeros((60, 60))
    x[30, 30] = 20.0  # un fuego chico de 20 MW
    nadir = observe(MODIS, "Aqua", 13.5, 0.0, x, rng, cloud_prob=0.0)
    edge = observe(MODIS, "Aqua", 13.5, 54.0, x, rng, cloud_prob=0.0)
    assert nadir.detected.any()
    assert not edge.detected.any()
    assert edge.thr_mw > 8 * nadir.thr_mw


def test_naive_gridding_preserves_detected_frp():
    rng = np.random.default_rng(2)
    x = np.zeros((40, 40))
    x[18:22, 18:22] = 30.0
    obs = observe(VIIRS, "S-NPP", 13.5, 10.0, x, rng, cloud_prob=0.0)
    assert grid_naive(obs, x.shape).sum() == pytest.approx(np.nansum(obs.frp))


def test_gaspari_cohn_shape():
    d = np.linspace(0, 3, 301)
    g = gaspari_cohn(d, 1.0)
    assert g[0] == pytest.approx(1.0)
    assert np.all(np.diff(g) <= 1e-12)
    assert np.all(g[d >= 2.0] == 0.0)


def test_model_energy_balance_and_diurnal_cycle():
    p = FireParams(sigma_proc=0.0)
    assert diurnal(14.0, p) == pytest.approx(1.0)
    assert diurnal(2.0, p) < 0.45
    x = np.zeros((30, 30))
    x[13:17, 13:17] = 15.0
    f = np.ones((30, 30))
    dt, energy = 1.0 / 6.0, 0.0
    for k in range(120):
        x, f = step(x, f, p, dt, 12.0)
        energy += x.sum() * dt
    assert x.min() >= 0 and f.min() >= 0 and f.max() <= 1
    assert f.min() < 0.05                      # el centro se quemó
    # toda la energía irradiada sale del combustible consumido
    assert energy == pytest.approx(p.e0 * (900 - f.sum()), rel=1e-6)


def test_intensity_is_bounded_by_fuel_and_time_of_day():
    """Una celda encendida tiende a x_peak * d(t) * f; nunca crece sin límite."""
    p = FireParams(sigma_proc=0.0, x_peak=20.0)
    x = np.full((20, 20), 5.0)
    f = np.full((20, 20), 0.5)
    peak = 0.0
    for k in range(60):
        x, f = step(x, f, p, 1.0 / 6.0, 14.0)
        peak = max(peak, x.max())
    assert 0.6 * 10.0 < peak <= 10.0           # equilibrio inicial: 20 MW x 1 x 0.5


def test_cold_cells_only_light_up_by_contagion():
    p = FireParams(sigma_proc=0.0)
    f = np.ones((30, 30))
    # un rescoldo aislado por debajo de x_off se enfría en vez de crecer
    x = np.zeros((30, 30))
    x[15, 15] = 0.5 * p.x_off
    for k in range(60):
        x, f = step(x, f, p, 1.0 / 6.0, 14.0)
    assert x.sum() < 0.1 * p.x_off
    # un parche encendido sí contagia a las celdas vecinas
    x = np.zeros((30, 30))
    x[13:17, 13:17] = 20.0
    for k in range(60):
        x, f = step(x, f, p, 1.0 / 6.0, 14.0)
    assert (x > p.x_off).sum() > 16 and x[5, 5] == 0.0


def test_step_is_vectorised_over_members():
    rng = np.random.default_rng(3)
    p = FireParams(sigma_proc=0.0).perturbed(4, rng)
    x = rng.gamma(1.0, 5.0, (4, 12, 12))
    f = np.full((4, 12, 12), 0.8)
    xa, fa = step(x, f, p, 0.1, 13.0)
    one = FireParams(beta=p.beta[2], x_peak=p.x_peak[2], tau_res=p.tau_res[2],
                     wind_x=p.wind_x[2], wind_y=p.wind_y[2], sigma_proc=0.0)
    xb, fb = step(x[2], f[2], one, 0.1, 13.0)
    np.testing.assert_allclose(xa[2], xb)
    np.testing.assert_allclose(fa[2], fb)


def test_letkf_matches_kalman_filter_on_linear_gaussian_problem():
    rng = np.random.default_rng(4)
    n, prior_sd, r = 400, 2.0, 0.5 ** 2
    ens = rng.standard_normal((n, 1))
    ens = 1.0 + prior_sd * (ens - ens.mean()) / ens.std(ddof=1)
    yo = np.array([4.0])
    letkf_update([ens], ens.copy(), yo, np.array([r]), np.zeros((1, 2)), np.zeros((1, 2)), 5.0)
    gain = prior_sd ** 2 / (prior_sd ** 2 + r)
    assert ens.mean() == pytest.approx(1.0 + gain * 3.0, abs=1e-8)
    assert ens.var(ddof=1) == pytest.approx((1 - gain) * prior_sd ** 2, rel=1e-8)


def test_letkf_localization_leaves_distant_cells_untouched():
    rng = np.random.default_rng(5)
    ens = rng.standard_normal((30, 2))
    before = ens.copy()
    cell_xy = np.array([[0.0, 0.0], [50.0, 0.0]])
    letkf_update([ens], ens[:, :1].copy(), np.array([3.0]), np.array([0.1]),
                 cell_xy, np.array([[0.0, 0.0]]), 1.0)
    assert not np.allclose(ens[:, 0], before[:, 0])
    np.testing.assert_array_equal(ens[:, 1], before[:, 1])
