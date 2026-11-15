from datetime import datetime, timedelta

import numpy as np
import pytest

from frp_enkf import aggregate as ag


def _hourly(start: datetime, hours: int):
    return [start + timedelta(hours=h) for h in range(hours)]


def test_period_keys_follow_local_calendar():
    w = datetime(2026, 11, 15, 23, 30)                      # domingo
    assert ag.period_key(w, "dia") == "2026-11-15"
    assert ag.period_key(w, "semana") == "2026-S46"
    assert ag.period_key(w + timedelta(hours=1), "semana") == "2026-S47"   # lunes
    assert ag.period_key(w, "mes") == "2026-11"
    local = ag.local_datetimes(np.array(["2026-11-08T18:00"]), 9.64)
    assert local[0].strftime("%Y-%m-%d %H") == "2026-11-09 03"            # ya es el día siguiente


def test_daily_weekly_monthly_energy_is_conserved():
    when = _hourly(datetime(2026, 10, 30), 24 * 40)        # 40 días, cruza dos meses
    frp = np.full((len(when), 3, 4), 2.0)                   # 2 MW por celda, constante
    total = frp.sum() * 1.0 / 1000.0
    for scale, n in (("dia", 40), ("semana", 7), ("mes", 3)):
        p = ag.aggregate(frp, when, scale)
        assert len(p.labels) == n
        assert p.total_gwh.sum() == pytest.approx(total)
    day = ag.aggregate(frp, when, "dia")
    assert day.complete.all() and day.energy_gwh[0, 0, 0] == pytest.approx(0.048)
    month = ag.aggregate(frp, when, "mes")
    assert month.labels == ["2026-10", "2026-11", "2026-12"]
    assert list(month.complete) == [False, True, False]    # octubre y diciembre, parciales
    assert month.full_hours[1] == 720.0


def test_peak_period_and_trend_show_a_moving_and_a_growing_fire():
    when = _hourly(datetime(2026, 11, 2), 24 * 5)
    frp = np.zeros((len(when), 2, 5))
    for d in range(5):
        frp[24 * d:24 * (d + 1), 0, d] = 10.0               # un foco que avanza una celda por día
        frp[24 * d:24 * (d + 1), 1, 0] = 2.0 * (d + 1)      # otro que crece en el lugar
    p = ag.aggregate(frp, when, "dia")
    peak = ag.peak_period(p)
    assert list(peak[0]) == [0, 1, 2, 3, 4]
    assert peak[1, 0] == 4 and peak[1, 3] == -1             # -1: nunca hubo fuego
    slope = ag.trend(p)
    assert slope[1, 0] == pytest.approx(2.0 * 24 / 1000.0)  # +2 MW por día
    assert slope[0, 0] < 0 < slope[0, 4]
    assert not ag.trend(ag.aggregate(frp[:48], when[:48], "dia")).any()   # 2 días: sin tendencia


def test_hourly_profile_and_detection_density():
    when = _hourly(datetime(2026, 11, 2), 48)
    frp = np.zeros((48, 1, 1))
    frp[[14, 38], 0, 0] = 30.0
    prof = ag.hourly_profile(frp, when)
    assert prof[14] == pytest.approx(30.0) and prof[3] == 0.0
    bbox = (10.0, -5.0, 12.0, -3.0)
    dens = ag.detection_density(np.array([-4.9, -4.9, -3.1]), np.array([10.1, 10.1, 11.9]),
                                [when[0], when[30], when[30]], "dia",
                                ["2026-11-02", "2026-11-03"], bbox, (4, 4))
    assert dens[0, 0, 0] == 1 and dens[1, 0, 0] == 1 and dens[1, 3, 3] == 1 and dens.sum() == 3
