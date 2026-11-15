"""Experimento de simulación de sistema de observación (OSSE).

1. Se simula un incendio "real" en la grilla de 375 m.
2. Se simulan las pasadas de Terra/Aqua (MODIS) y S-NPP/NOAA-20 (VIIRS).
3. El filtro, que arranca sin saber nada del fuego y con un modelo imperfecto,
   asimila las pasadas. Se compara contra la verdad y contra una línea de base
   que solo grilla la última pasada disponible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assim import FilterConfig, FireFilter, RunResult, run_passes
from .model import FireParams, make_fuel_map, smooth_field, step
from .sensors import MODIS, VIIRS, Observation, Sensor, grid_naive, observe

# (plataforma, sensor, horas locales de pasada)
PLATFORMS: list[tuple[str, Sensor, tuple[float, float]]] = [
    ("Terra", MODIS, (10.5, 22.5)),
    ("Aqua", MODIS, (13.5, 1.5)),
    ("S-NPP", VIIRS, (13.67, 1.67)),
    ("NOAA-20", VIIRS, (12.83, 0.83)),
]


@dataclass
class Scenario:
    shape: tuple[int, int] = (96, 96)
    t0_h: float = 6.0            # hora local de inicio
    hours: float = 72.0
    dt_h: float = 1.0 / 6.0      # 10 minutos
    ignition_h: float = 10.0
    ignition_cell: tuple[int, int] | None = None   # None: la celda con más combustible del centro
    ignition_mw: float = 15.0   # por celda, en un parche de 5 x 5
    seed: int = 7
    truth: FireParams = field(default_factory=lambda: FireParams(
        beta=0.40, x_peak=40.0, tau_res=4.0, wind_x=0.3, wind_y=0.1))
    # el modelo del filtro está deliberadamente mal calibrado
    model: FireParams = field(default_factory=lambda: FireParams(
        beta=0.34, x_peak=46.0, tau_res=3.4, d_min=0.45, peak_hour=13.0,
        wind_x=0.1, wind_y=0.25))
    fuel_error: float = 0.12     # error del mapa de combustible del filtro
    cloud_prob: float = 0.3

    @property
    def n_steps(self) -> int:
        return int(round(self.hours / self.dt_h))


@dataclass
class Truth:
    times: np.ndarray            # (T+1,) horas desde la medianoche del día 0
    x: np.ndarray                # (T+1, ny, nx)
    fuel0: np.ndarray
    passes: dict[int, list[Observation]]   # paso -> pasadas que caen en ese paso


def _richest_central_cell(fuel: np.ndarray) -> tuple[int, int]:
    """Celda con más combustible en el tercio central de la grilla."""
    ny, nx = fuel.shape
    sub = fuel[ny // 3:2 * ny // 3, nx // 3:2 * nx // 3]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return int(iy + ny // 3), int(ix + nx // 3)


def simulate_truth(sc: Scenario) -> Truth:
    rng = np.random.default_rng(sc.seed)
    fuel0 = make_fuel_map(sc.shape, rng)
    x, f = np.zeros(sc.shape), fuel0.copy()
    times = sc.t0_h + sc.dt_h * np.arange(sc.n_steps + 1)
    xs = np.zeros((sc.n_steps + 1,) + sc.shape)
    for k in range(sc.n_steps):
        if abs(times[k] - sc.ignition_h) < 0.5 * sc.dt_h:
            iy, ix = sc.ignition_cell or _richest_central_cell(fuel0)
            x[iy - 2:iy + 3, ix - 2:ix + 3] = sc.ignition_mw
        x, f = step(x, f, sc.truth, sc.dt_h, times[k], rng)
        xs[k + 1] = x

    passes: dict[int, list[Observation]] = {}
    n_days = int(np.ceil((sc.t0_h + sc.hours) / 24.0)) + 1
    schedule = []
    for day in range(n_days):
        for platform, sensor, hours in PLATFORMS:
            for h in hours:
                t = 24.0 * day + h + rng.uniform(-0.4, 0.4)
                theta = rng.uniform(-sensor.max_scan_deg, sensor.max_scan_deg)
                schedule.append((t, platform, sensor, theta))
    for t, platform, sensor, theta in sorted(schedule, key=lambda s: s[0]):
        k = int(round((t - sc.t0_h) / sc.dt_h))
        if 1 <= k <= sc.n_steps:
            obs = observe(sensor, platform, float(times[k]), theta, xs[k], rng, sc.cloud_prob)
            passes.setdefault(k, []).append(obs)
    return Truth(times, xs, fuel0, passes)


def run_filter(sc: Scenario, truth: Truth, families: tuple[str, ...], label: str,
               n_members: int = 50, cfg: FilterConfig | None = None,
               seed: int = 100) -> RunResult:
    """Corre el filtro asimilando solo las familias de sensores indicadas."""
    rng = np.random.default_rng(seed)
    cfg = cfg or FilterConfig()
    if "VIIRS" not in families:
        cfg = FilterConfig(**{**cfg.__dict__, "estimate_bias": False})
    fuel = np.clip(truth.fuel0 + sc.fuel_error * smooth_field(sc.shape, 4.0, rng, n_members),
                   0.2, 1.0)
    flt = FireFilter(fuel, sc.model.perturbed(n_members, rng), rng, cfg)
    passes = {k: [o for o in ps if o.sensor.family in families]
              for k, ps in truth.passes.items()}
    return run_passes(flt, passes, truth.times, label)


def run_naive(sc: Scenario, truth: Truth, families: tuple[str, ...], label: str) -> RunResult:
    """Línea de base: mapa grillado de la última pasada, mantenido hasta la siguiente."""
    mean = np.zeros((sc.n_steps + 1,) + sc.shape)
    current = np.zeros(sc.shape)
    for k in range(1, sc.n_steps + 1):
        for obs in truth.passes.get(k, []):
            if obs.sensor.family in families:
                current = grid_naive(obs, sc.shape)
        mean[k] = current
    return RunResult(label, mean, None, None, [], [])


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Promedia bloques de k x k celdas (últimas dos dimensiones)."""
    ny, nx = a.shape[-2] // k * k, a.shape[-1] // k * k
    a = a[..., :ny, :nx]
    return a.reshape(a.shape[:-2] + (ny // k, k, nx // k, k)).mean(axis=(-3, -1))


def score(sc: Scenario, truth: Truth, res: RunResult, t_start_h: float) -> dict:
    """Métricas contra la verdad desde t_start_h (para excluir el período sin datos)."""
    sel = truth.times >= t_start_h
    err = res.mean[sel] - truth.x[sel]
    tot_t = truth.x[sel].sum(axis=(1, 2))
    tot_m = res.mean[sel].sum(axis=(1, 2))
    out = {
        "rmse_375m": float(np.sqrt((err ** 2).mean())),
        "rmse_1500m": float(np.sqrt((block_mean(err, 4) ** 2).mean())),
        "total_mae_pct": float(100 * np.abs(tot_m - tot_t).mean() / tot_t.mean()),
        "total_bias_pct": float(100 * (tot_m - tot_t).mean() / tot_t.mean()),
    }
    if res.spread is not None:
        burning = (truth.x[sel] > 0.5) | (res.mean[sel] > 0.5)
        out["spread_over_rmse"] = float(
            np.sqrt((res.spread[sel][burning] ** 2).mean()) / np.sqrt((err[burning] ** 2).mean()))
    return out
