"""Sensores: geometría de píxel, operador de observación H y simulación de pasadas.

Cada pasada tiene su propio H (matriz dispersa píxeles x celdas finas) construido
a partir de la huella real del píxel para el ángulo de barrido de esa pasada:

* MODIS: 1 km en nadir, hasta ~4.8 km x 2.0 km en el borde; respuesta triangular
  a lo largo del barrido (píxeles vecinos se solapan y la energía se conserva).
* VIIRS banda I: 375 m en nadir con agregación de muestras 3:1 / 2:1 / 1:1, lo que
  limita el crecimiento a ~0.8 km en el borde; respuesta rectangular.

La detección usa un umbral proporcional al área del píxel: es lo que hace que
MODIS pierda los fuegos chicos lejos del nadir.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .model import CELL_KM, smooth_field

R_EARTH = 6371.0


@dataclass(frozen=True)
class Sensor:
    name: str
    family: str                # "MODIS" o "VIIRS"
    altitude_km: float
    nadir_scan_km: float       # tamaño de la muestra nativa a lo largo del barrido
    nadir_track_km: float
    max_scan_deg: float
    triangular_psf: bool
    thr_density: float         # umbral de detección [MW por km2 de píxel]
    sigma_log: float           # ruido multiplicativo del FRP (desvío en log)
    bias: float = 1.0          # sesgo multiplicativo real (solo para simular)
    sat_mw: float | None = None  # saturación opcional del FRP por píxel
    # zonas de agregación: (ángulo máximo en grados, muestras agregadas)
    agg_zones: tuple[tuple[float, int], ...] = ((90.0, 1),)

    def pixel_size(self, theta_deg: float) -> tuple[float, float]:
        """Tamaño del píxel (barrido, trayectoria) en km para el ángulo de barrido."""
        th = np.deg2rad(abs(theta_deg))
        r = R_EARTH + self.altitude_km
        root = np.sqrt((R_EARTH / r) ** 2 - np.sin(th) ** 2)
        ds = R_EARTH * (self.nadir_scan_km / self.altitude_km) * (np.cos(th) / root - 1.0)
        dt = r * (self.nadir_track_km / self.altitude_km) * (np.cos(th) - root)
        n_agg = next(n for lim, n in self.agg_zones if abs(theta_deg) <= lim)
        return float(ds * n_agg), float(dt)


MODIS = Sensor("MODIS", "MODIS", 705.0, 1.0, 1.0, 55.0, True,
               thr_density=7.0, sigma_log=0.20, bias=0.80)
VIIRS = Sensor("VIIRS", "VIIRS", 829.0, 0.125, 0.375, 56.0, False,
               thr_density=7.0, sigma_log=0.12,
               agg_zones=((31.72, 3), (44.86, 2), (90.0, 1)))


@dataclass
class Observation:
    """Una pasada: huellas, máscara de nubes, detecciones y FRP por píxel."""

    sensor: Sensor
    platform: str
    time_h: float
    theta_deg: float
    H: sparse.csr_matrix       # (n_pix, n_celdas): fracción de cada celda vista
    centers_km: np.ndarray     # (n_pix, 2) centro (x, y) de cada píxel
    ds_km: float
    dt_km: float
    thr_mw: float              # umbral de detección de esta pasada
    clear: np.ndarray          # (n_pix,) bool: sin nubes
    detected: np.ndarray       # (n_pix,) bool
    frp: np.ndarray            # (n_pix,) MW; NaN donde no hay detección

    @property
    def area_km2(self) -> float:
        return self.ds_km * self.dt_km


def build_footprints(sensor: Sensor, theta_deg: float, heading_deg: float,
                     shape: tuple[int, int], offset: tuple[float, float] = (0.0, 0.0),
                     nsub: int = 4) -> tuple[sparse.csr_matrix, np.ndarray, float, float]:
    """Construye H para una pasada.

    Devuelve (H, centros, ds, dt). H[p, c] es la fracción de la energía de la celda
    c que cae en el píxel p; para cada celda las fracciones suman 1.
    """
    ny, nx = shape
    ds, dt = sensor.pixel_size(theta_deg)
    phi = np.deg2rad(heading_deg)
    cs, sn = np.cos(phi), np.sin(phi)

    sub = (np.arange(nsub) + 0.5) / nsub
    yy, xx, sy, sx = np.meshgrid(np.arange(ny), np.arange(nx), sub, sub, indexing="ij")
    x = ((xx + sx) * CELL_KM).ravel()
    y = ((yy + sy) * CELL_KM).ravel()
    cell = (yy * nx + xx).ravel()
    u = (x * cs + y * sn) / ds + offset[0]
    v = (-x * sn + y * cs) / dt + offset[1]
    iv = np.floor(v).astype(np.int64)
    w0 = np.full(u.shape, 1.0 / nsub ** 2)
    if sensor.triangular_psf:
        iu0 = np.floor(u - 0.5).astype(np.int64)
        fr = u - 0.5 - iu0
        iu = np.concatenate([iu0, iu0 + 1])
        iv = np.concatenate([iv, iv])
        w = np.concatenate([w0 * (1 - fr), w0 * fr])
        cell = np.concatenate([cell, cell])
    else:
        iu, w = np.floor(u).astype(np.int64), w0

    iu -= iu.min()
    iv -= iv.min()
    nu = iu.max() + 1
    key = iv * nu + iu
    uniq, pix = np.unique(key, return_inverse=True)
    H = sparse.coo_matrix((w, (pix, cell)), shape=(uniq.size, ny * nx)).tocsr()
    H.sum_duplicates()

    # centro geométrico de cada píxel (centroide ponderado de sus celdas)
    cx = (np.arange(ny * nx) % nx + 0.5) * CELL_KM
    cy = (np.arange(ny * nx) // nx + 0.5) * CELL_KM
    tot = np.asarray(H.sum(axis=1)).ravel()
    centers = np.column_stack([H @ cx / tot, H @ cy / tot])
    return H, centers, ds, dt


def observe(sensor: Sensor, platform: str, time_h: float, theta_deg: float,
            x_true: np.ndarray, rng: np.random.Generator,
            cloud_prob: float = 0.3) -> Observation:
    """Simula lo que el sensor reporta sobre el campo real x_true (ny, nx)."""
    shape = x_true.shape
    H, centers, ds, dt = build_footprints(
        sensor, theta_deg, heading_deg=rng.uniform(5.0, 15.0), shape=shape,
        offset=(rng.uniform(), rng.uniform()))
    thr = sensor.thr_density * ds * dt

    y = sensor.bias * (H @ x_true.ravel())
    y = y * np.exp(sensor.sigma_log * rng.standard_normal(y.size))
    y = y + 0.1 * thr * rng.standard_normal(y.size)
    if sensor.sat_mw is not None:
        y = np.minimum(y, sensor.sat_mw)

    clear = np.ones(y.size, dtype=bool)
    if rng.uniform() < cloud_prob:
        field = smooth_field(shape, 6.0, rng).ravel()
        cloudy_cells = (field > np.quantile(field, 1 - rng.uniform(0.05, 0.35))).astype(float)
        frac = (H @ cloudy_cells) / np.asarray(H.sum(axis=1)).ravel()
        clear = frac < 0.5

    detected = clear & (y > thr)
    return Observation(sensor, platform, time_h, theta_deg, H, centers, ds, dt, thr,
                       clear, detected, np.where(detected, y, np.nan))


def grid_naive(obs: Observation, shape: tuple[int, int]) -> np.ndarray:
    """Línea de base: reparte el FRP de cada detección uniformemente en su huella."""
    frp = np.where(obs.detected, obs.frp, 0.0)
    tot = np.asarray(obs.H.sum(axis=1)).ravel()
    return (obs.H.T @ (frp / tot)).reshape(shape)
