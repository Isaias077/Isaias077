import numpy as np

from frp_enkf.assim import FilterConfig, FireFilter
from frp_enkf.model import FireParams
from frp_enkf.osse import Scenario, run_filter, run_naive, score, simulate_truth
from frp_enkf.sensors import MODIS, VIIRS, observe


def _filter(n=30, shape=(40, 40), seed=0, **cfg):
    rng = np.random.default_rng(seed)
    return FireFilter(np.full((n,) + shape, 0.8), FireParams().perturbed(n, rng), rng,
                      FilterConfig(**cfg))


def test_detection_ignites_a_blind_ensemble_inside_the_footprint():
    truth = np.zeros((40, 40))
    truth[20:23, 20:23] = 25.0
    flt = _filter()
    obs = observe(VIIRS, "S-NPP", 13.5, 5.0, truth, np.random.default_rng(1), cloud_prob=0.0)
    info = flt.assimilate(obs)
    assert info["born"] > 0
    mean = flt.x.mean(axis=0)
    np.testing.assert_allclose(mean.sum(), truth.sum(), rtol=0.2)
    assert mean[15:28, 15:28].sum() > 0.98 * mean.sum()


def test_clear_non_detection_puts_out_a_false_fire():
    flt = _filter()
    flt.x[:, 10:13, 10:13] = 20.0 * np.exp(flt.rng.standard_normal((30, 3, 3)))
    before = flt.x.mean(axis=0).sum()
    obs = observe(VIIRS, "S-NPP", 13.5, 5.0, np.zeros((40, 40)), np.random.default_rng(2),
                  cloud_prob=0.0)
    flt.assimilate(obs)
    assert flt.x.mean(axis=0).sum() < 0.25 * before


def test_cloudy_pixels_carry_no_information():
    flt = _filter()
    flt.x[:, 10:13, 10:13] = 20.0 * np.exp(0.3 * flt.rng.standard_normal((30, 3, 3)))
    before = flt.x.copy()
    obs = observe(VIIRS, "S-NPP", 13.5, 5.0, np.zeros((40, 40)), np.random.default_rng(2),
                  cloud_prob=0.0)
    obs.clear[:] = False
    flt.assimilate(obs)
    np.testing.assert_allclose(flt.x, before, rtol=1e-12)


def test_modis_pixel_is_shared_out_where_viirs_saw_fire():
    """El FRP de un píxel MODIS grande va a las celdas que VIIRS ya había marcado."""
    truth = np.zeros((40, 40))
    truth[18:20, 24:26] = 40.0
    flt = _filter(estimate_bias=False)
    flt.assimilate(observe(VIIRS, "S-NPP", 13.4, 0.0, truth, np.random.default_rng(3), 0.0))
    # pasan horas sin datos: la intensidad de cada miembro se vuelve incierta
    flt.x *= np.exp(0.5 * flt.rng.standard_normal((flt.n, 1, 1)))
    before = flt.x.mean(axis=0).sum()
    modis = MODIS.__class__(**{**MODIS.__dict__, "bias": 1.0})
    obs = observe(modis, "Aqua", 13.5, 50.0, truth * 1.5, np.random.default_rng(4), 0.0)
    assert obs.detected.any() and obs.ds_km > 3.0
    flt.assimilate(obs)
    mean = flt.x.mean(axis=0)
    assert mean[18:20, 24:26].sum() > 0.8 * mean.sum()          # no se desparrama
    assert abs(mean.sum() - 1.5 * truth.sum()) < abs(before - 1.5 * truth.sum())  # y se acerca a MODIS


def test_osse_filter_beats_naive_baseline_and_recovers_bias():
    sc = Scenario(shape=(64, 64), hours=56.0, seed=7)
    truth = simulate_truth(sc)
    t_start = truth.times[min(truth.passes)] + 6.0
    fam = ("MODIS", "VIIRS")
    res = run_filter(sc, truth, fam, "filtro", n_members=30)
    s_f = score(sc, truth, res, t_start)
    s_n = score(sc, truth, run_naive(sc, truth, fam, "naive"), t_start)
    assert s_f["rmse_375m"] < 0.75 * s_n["rmse_375m"]
    assert s_f["rmse_1500m"] < 0.75 * s_n["rmse_1500m"]
    assert s_f["total_mae_pct"] < s_n["total_mae_pct"]
    # el sesgo estimado es compatible con el real dentro de su propia incertidumbre
    _, b_hat, log_sd = res.bias_trace[-1]
    assert log_sd < FilterConfig().bias_prior_sd
    assert abs(np.log(b_hat / MODIS.bias)) < 2.5 * log_sd


def test_new_fire_gets_the_fuel_that_explains_its_intensity():
    """Una detección débil no debe convertirse en un fuego a pleno en la hora siguiente."""
    truth = np.zeros((40, 40))
    truth[20:23, 20:23] = 4.0
    flt = _filter()
    obs = observe(VIIRS, "S-NPP", 13.5, 5.0, truth, np.random.default_rng(1), cloud_prob=0.0)
    flt.assimilate(obs)
    start = flt.x.mean(axis=0).sum()
    for k in range(6):
        flt.forecast(1.0 / 6.0, 13.5 + k / 6.0)
    assert flt.x.mean(axis=0).sum() < 2.0 * start
    assert flt.f[:, 20:23, 20:23].mean() < 0.4


def test_online_parameter_estimation_lowers_rates_when_forecast_overshoots():
    flt = _filter(estimate_params=True)
    flt.n_assimilated = 1
    truth = np.zeros((40, 40))
    truth[20:23, 20:23] = 5.0
    # los miembros con x_peak más alto pronostican más FRP: todos se pasan
    flt.x[:, 20:23, 20:23] = (flt.params.x_peak * 0.6)[:, None, None]
    before = np.exp(np.log(flt.params.x_peak).mean())
    obs = observe(VIIRS, "S-NPP", 13.5, 5.0, truth, np.random.default_rng(1), cloud_prob=0.0)
    flt.assimilate(obs)
    assert np.exp(np.log(flt.params.x_peak).mean()) < 0.9 * before


def test_holdout_pass_is_verified_but_not_assimilated():
    from frp_enkf.assim import run_passes
    truth = np.zeros((40, 40))
    truth[20:23, 20:23] = 25.0
    flt = _filter()
    times = 13.0 + np.arange(4) / 6.0
    seen = observe(VIIRS, "NOAA-20", 13.2, 5.0, truth, np.random.default_rng(1), cloud_prob=0.0)
    held = observe(VIIRS, "NOAA-21", 13.5, 20.0, truth, np.random.default_rng(2), cloud_prob=0.0)
    res = run_passes(flt, {1: [seen]}, times, "x", holdout={3: [held]})
    assert len(res.log) == 1 and len(res.verification) == 1
    v = res.verification[0]
    assert v["platform"] == "NOAA-21" and v["n_detected"] == held.detected.sum()
    assert v["filtro_hits"] + v["filtro_misses"] == v["n_detected"]
    assert 0.5 * v["obs_total"] < v["filtro_total"] < 2.0 * v["obs_total"]
