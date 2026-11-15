"""Calibración del modelo de fuego con las propias detecciones de FIRMS.

Dos parámetros salen directo de los datos, sin correr el filtro:

* x_peak, el FRP de una celda de 375 m a pleno: percentil 90 del FRP de los
  píxeles VIIRS diurnos cercanos al nadir (los que miden una sola celda).
* d_min, el piso nocturno del ciclo diurno: cociente entre la mediana nocturna y
  la diurna de esos mismos píxeles.

La velocidad de contagio y el tiempo de residencia no se ven en una detección
suelta; esos se ajustan en línea durante la asimilación.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model import CELL_KM, FireParams

NEAR_NADIR_KM2 = 1.5 * CELL_KM ** 2   # hasta 1.5 celdas de área
MIN_SAMPLES = 30


def calibrate_from_detections(dets: list, base: FireParams | None = None
                              ) -> tuple[FireParams, dict]:
    """Devuelve (parámetros calibrados, informe). Sin datos suficientes deja `base`."""
    base = base or FireParams()
    frp, day = [], []
    for d in dets:
        if d.sensor.family != "VIIRS" or not len(d):
            continue
        near = d.scan * d.track <= NEAR_NADIR_KM2
        frp.append(d.frp[near])
        day.append(d.day[near])
    frp = np.concatenate(frp) if frp else np.array([])
    day = np.concatenate(day) if day else np.array([], dtype=bool)
    report = {"n_dia": int(day.sum()), "n_noche": int((~day).sum())}
    out = replace(base)
    if day.sum() >= MIN_SAMPLES:
        out.x_peak = float(np.percentile(frp[day], 90))
        report["x_peak_mw"] = out.x_peak
        if (~day).sum() >= MIN_SAMPLES:
            out.d_min = float(np.clip(np.median(frp[~day]) / np.median(frp[day]), 0.15, 0.8))
            report["d_min"] = out.d_min
    return out, report
