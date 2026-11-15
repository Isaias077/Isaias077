"""Modelo estocástico de fuego sobre una grilla fina (375 m).

Estado por celda:
    x : potencia radiativa del fuego, FRP [MW]
    f : fracción de combustible restante [0, 1]

Una celda está encendida si x > x_off. Dinámica (paso dt en horas):

    encendida:  x' = (x_eq - x) / tau_x,   x_eq = x_peak * d(t) * f
    apagada:    x' = beta * d(t) * f * s * s / (s + x_ign) - lam * x,   s = K * x
    siempre:    f' = -x / (x_peak * tau_res)

La intensidad de una celda encendida no crece sin límite: tiende a un equilibrio
fijado por su combustible y la hora del día, con x_peak como techo. Al consumir
combustible (en unas tau_res horas a pleno) el equilibrio baja hasta que la celda
se apaga. Una celda apagada solo se enciende por contagio: acumula el calor s que
le llega de sus vecinas (K es un núcleo 3x3 desplazado por el viento) hasta
superar x_off, así que beta fija la velocidad del frente. d(t) es el ciclo diurno.

Como el FRP es proporcional a la tasa de consumo de biomasa, la energía irradiada
sale siempre del combustible: x_peak * tau_res es la energía total de una celda.

Todas las funciones aceptan arreglos con dimensiones delanteras de ensemble:
x y f con forma (..., ny, nx) y parámetros con forma (...,).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

CELL_KM = 0.375
X_MIN = 1e-3  # MW; por debajo de esto la celda se considera apagada

_OFFSETS = [(oy, ox) for oy in (-1, 0, 1) for ox in (-1, 0, 1)]


@dataclass
class FireParams:
    """Parámetros del modelo. beta, x_peak, tau_res y viento: escalar o arreglo (N,)."""

    beta: np.ndarray | float = 0.3     # contagio a celdas vecinas [1/h]
    x_peak: np.ndarray | float = 40.0  # FRP de una celda a pleno (f = 1, mediodía) [MW]
    tau_res: np.ndarray | float = 4.0  # horas que tarda una celda a pleno en consumirse
    wind_x: np.ndarray | float = 0.0   # desplazamiento del núcleo [celdas]
    wind_y: np.ndarray | float = 0.0
    tau_x: float = 0.5                 # tiempo de respuesta de la intensidad [h]
    lam: float = 0.2                   # enfriamiento de una celda apagada [1/h]
    x_ign: float = 2.0                 # umbral suave del calor recibido [MW]
    x_off: float = 1.0                 # por debajo de esto la celda está apagada [MW]
    sigma_proc: float = 0.25           # ruido de proceso multiplicativo [1/sqrt(h)]
    d_min: float = 0.4                 # piso nocturno del ciclo diurno
    peak_hour: float = 14.0            # hora local del máximo diurno
    sigma_hour: float = 3.5            # ancho del máximo diurno [h]

    RATES = ("beta", "x_peak", "tau_res")   # los que se perturban y se pueden estimar

    @property
    def e0(self) -> np.ndarray | float:
        """Energía radiativa total de una celda con combustible completo [MW h]."""
        return np.asarray(self.x_peak) * np.asarray(self.tau_res)

    def perturbed(self, n: int, rng: np.random.Generator, rel: float = 0.15,
                  wind_sd: float = 0.15) -> "FireParams":
        """N juegos de parámetros alrededor de los centrales (para el ensemble)."""
        out = replace(self)
        for name in self.RATES:
            setattr(out, name, float(getattr(self, name)) * np.exp(rel * rng.standard_normal(n)))
        out.wind_x = float(self.wind_x) + wind_sd * rng.standard_normal(n)
        out.wind_y = float(self.wind_y) + wind_sd * rng.standard_normal(n)
        return out


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
    noise = 1.0
    if rng is not None and p.sigma_proc > 0:
        s = p.sigma_proc * np.sqrt(dt)
        noise = np.exp(s * rng.standard_normal(x.shape) - 0.5 * s * s)
    heat = _spread(x, _kernel_weights(p))
    # encendida: relaja hacia el equilibrio que permite su combustible
    x_eq = col(p.x_peak) * d * f * noise
    x_on = x + (x_eq - x) * (1.0 - np.exp(-dt / p.tau_x))
    # apagada: acumula el calor de las vecinas (y lo pierde despacio)
    x_cold = x * np.exp(-p.lam * dt) + col(p.beta) * d * f * heat * heat / (heat + p.x_ign) * dt * noise
    x_new = np.where(x > p.x_off, x_on, x_cold)
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
