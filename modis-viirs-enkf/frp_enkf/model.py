"""Modelo estocástico de fuego sobre una grilla fina (375 m).

Estado por celda:
    x : potencia radiativa del fuego, FRP [MW]
    f : fracción de combustible restante [0, 1]

Dinámica (Euler, paso dt en horas):
    s  = K * x                                  (calor que llega de los vecinos)
    x' = d(t) * f * (beta_self * x + beta * s * s / (s + x_ign)) - lam * x
    f' = -x / e0

K es un núcleo 3x3 desplazado por el viento, d(t) el ciclo diurno y e0 la
energía radiativa total que libera una celda con combustible completo [MW h].
El término beta_self hace que una celda encendida siga ardiendo mientras le quede
combustible (de noche d(t) baja y el fuego se debilita sin apagarse). El factor
s / (s + x_ign) es un umbral suave de ignición: el calor débil no enciende celdas
nuevas, así que el fuego avanza como un frente y no se difunde.
Como FRP es proporcional a la tasa de consumo de biomasa, el combustible baja
en proporción a x: eso produce un frente que avanza y deja cicatriz quemada.

Todas las funciones aceptan arreglos con dimensiones delanteras de ensemble:
x y f con forma (..., ny, nx) y parámetros con forma (...,).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CELL_KM = 0.375
X_MIN = 1e-3  # MW; por debajo de esto la celda se considera apagada

_OFFSETS = [(oy, ox) for oy in (-1, 0, 1) for ox in (-1, 0, 1)]


@dataclass
class FireParams:
    """Parámetros del modelo. Cada campo es escalar o un arreglo (N,)."""

    beta: np.ndarray | float = 2.5   # tasa de propagación a vecinos [1/h]
    beta_self: np.ndarray | float = 1.0  # tasa de intensificación en la celda [1/h]
    lam: np.ndarray | float = 0.5    # tasa de decaimiento [1/h]
    e0: np.ndarray | float = 60.0    # energía por celda con f=1 [MW h]
    wind_x: np.ndarray | float = 0.0  # desplazamiento del núcleo [celdas]
    wind_y: np.ndarray | float = 0.0
    x_ign: float = 2.0               # umbral suave de ignición [MW]
    tau_burn: float = 0.5            # tiempo mínimo de agotamiento del combustible [h]
    sigma_proc: float = 0.25         # ruido de proceso multiplicativo [1/sqrt(h)]
    d_min: float = 0.4               # piso nocturno del ciclo diurno
    peak_hour: float = 14.0          # hora local del máximo diurno
    sigma_hour: float = 3.5          # ancho del máximo diurno [h]

    def perturbed(self, n: int, rng: np.random.Generator, rel: float = 0.15,
                  wind_sd: float = 0.15) -> "FireParams":
        """N juegos de parámetros alrededor de los centrales (para el ensemble)."""
        ln = lambda v, s: float(v) * np.exp(s * rng.standard_normal(n))
        return FireParams(
            beta=ln(self.beta, rel), beta_self=ln(self.beta_self, rel), lam=ln(self.lam, rel), e0=ln(self.e0, rel),
            wind_x=float(self.wind_x) + wind_sd * rng.standard_normal(n),
            wind_y=float(self.wind_y) + wind_sd * rng.standard_normal(n),
            x_ign=self.x_ign, tau_burn=self.tau_burn, sigma_proc=self.sigma_proc, d_min=self.d_min,
            peak_hour=self.peak_hour, sigma_hour=self.sigma_hour,
        )


def diurnal(hour: float, p: FireParams) -> float:
    """Factor diurno en [d_min, 1] para la hora local `hour` (se envuelve en 24 h)."""
    dh = (hour - p.peak_hour + 12.0) % 24.0 - 12.0
    return p.d_min + (1.0 - p.d_min) * float(np.exp(-0.5 * (dh / p.sigma_hour) ** 2))


def _kernel_weights(p: FireParams) -> np.ndarray:
    """Pesos del núcleo 3x3 desplazado por el viento. Forma (..., 9), suma 1."""
    wx = np.asarray(p.wind_x, dtype=float)[..., None]
    wy = np.asarray(p.wind_y, dtype=float)[..., None]
    oy = np.array([o[0] for o in _OFFSETS], dtype=float)
    ox = np.array([o[1] for o in _OFFSETS], dtype=float)
    w = np.exp(-((ox - wx) ** 2 + (oy - wy) ** 2) / (2 * 0.8 ** 2))
    return w / w.sum(axis=-1, keepdims=True)


def _spread(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Convolución K*x con bordes en cero. El offset o lleva energía de c-o a c."""
    ny, nx = x.shape[-2:]
    pad = np.zeros(x.shape[:-2] + (ny + 2, nx + 2))
    pad[..., 1:-1, 1:-1] = x
    out = np.zeros_like(x)
    for k, (oy, ox) in enumerate(_OFFSETS):
        out += w[..., k, None, None] * pad[..., 1 - oy:1 - oy + ny, 1 - ox:1 - ox + nx]
    return out


def step(x: np.ndarray, f: np.ndarray, p: FireParams, dt: float, hour: float,
         rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Avanza (x, f) un paso dt [h]. Con rng aplica ruido de proceso lognormal."""
    col = lambda v: np.asarray(v, dtype=float)[..., None, None]
    d = diurnal(hour, p)
    heat = _spread(x, _kernel_weights(p))
    gain = d * f * (col(p.beta_self) * x + col(p.beta) * heat * heat / (heat + p.x_ign)) * dt
    if rng is not None and p.sigma_proc > 0:
        s = p.sigma_proc * np.sqrt(dt)
        gain = gain * np.exp(s * rng.standard_normal(x.shape) - 0.5 * s * s)
    x_new = x * np.exp(-col(p.lam) * dt) + gain
    # una celda no puede irradiar más rápido de lo que le permite su combustible:
    # como mucho lo agota con tiempo característico tau_burn (conserva la energía)
    x_new = np.minimum(x_new, f * col(p.e0) / p.tau_burn)
    x_new = np.where(x_new < X_MIN, 0.0, x_new)
    f_new = np.clip(f - x_new * dt / col(p.e0), 0.0, 1.0)
    return x_new, f_new


def smooth_field(shape: tuple[int, int], length_cells: float,
                 rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """Campo aleatorio gaussiano suave (media 0, desvío 1) por filtrado en Fourier."""
    ny, nx = shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    filt = np.exp(-2 * (np.pi * length_cells) ** 2 * (kx ** 2 + ky ** 2))
    white = rng.standard_normal((n or 1, ny, nx))
    out = np.fft.ifft2(np.fft.fft2(white) * filt).real
    out /= out.std(axis=(-2, -1), keepdims=True)
    return out if n is not None else out[0]


def make_fuel_map(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Mapa de combustible heterogéneo en [0.35, 1]."""
    g = smooth_field(shape, 5.0, rng)
    return np.clip(0.75 + 0.2 * g, 0.35, 1.0)
