"""Filtro de fusión: ensemble de fuegos + LETKF con un H distinto por sensor.

El análisis se hace en espacio logarítmico (z = log(x + eps)) porque el FRP es
positivo y muy asimétrico. Las no-detecciones se asimilan como observaciones
censuradas ("FRP por debajo del umbral") y las detecciones que ningún miembro
explica encienden fuego nuevo dentro de la huella del píxel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .letkf import letkf_update
from .model import CELL_KM, X_MIN, FireParams, diurnal, step
from .sensors import Observation, grid_naive


@dataclass
class FilterConfig:
    eps: float = 0.1            # MW; offset de la transformación logarítmica
    loc_km: float = 1.0         # semiancho base de localización (se suma media huella)
    infl: float = 1.03          # inflación multiplicativa
    sigma_nondet: float = 0.3   # error (log) de una no-detección
    nondet_level: float = 1.0   # una no-detección despejada significa "FRP <= nivel * umbral"
    birth_frac: float = 0.1     # un miembro "no explica" si predice < frac * obs
    estimate_bias: bool = True  # estimar en línea el sesgo MODIS/VIIRS
    bias_prior_sd: float = 0.3  # desvío a priori de log(b)
    bias_walk_sd: float = 0.02  # caminata aleatoria de log(b) por pasada
    bias_drift_per_h: float = 0.08  # desconfianza por hora desde el último VIIRS
    birth_fuel_weight: float = 0.5  # 0: combustible a priori; 1: el que explica la intensidad
    estimate_params: bool = False   # ajustar en línea las tasas del modelo de fuego
    param_obs_sd: float = 0.3       # error (log) del FRP total usado para ajustarlas
    param_min_sd: float = 0.05      # dispersión mínima (log) que conservan las tasas


class FireFilter:
    def __init__(self, fuel: np.ndarray, params: FireParams,
                 rng: np.random.Generator, cfg: FilterConfig | None = None):
        """fuel: (N, ny, nx) combustible inicial de cada miembro; params: (N,)."""
        self.cfg = cfg or FilterConfig()
        self.rng = rng
        self.params = params
        self.f = fuel.copy()
        self.x = np.zeros_like(fuel)
        self.n, self.ny, self.nx = fuel.shape
        ii = np.arange(self.ny * self.nx)
        self.cell_xy = np.column_stack([(ii % self.nx + 0.5) * CELL_KM,
                                        (ii // self.nx + 0.5) * CELL_KM])
        self.log_b = 0.0                      # log del sesgo MODIS respecto de VIIRS
        self.log_b_var = self.cfg.bias_prior_sd ** 2
        self.last_viirs_h: float | None = None
        self.n_assimilated = 0
        self.viirs_surprise = np.inf          # |innovación log media| del último VIIRS

    # ------------------------------------------------------------------ pronóstico
    def forecast(self, dt: float, hour: float) -> None:
        self.x, self.f = step(self.x, self.f, self.params, dt, hour, self.rng)

    # ------------------------------------------------------------------ análisis
    def _hx(self, obs: Observation) -> np.ndarray:
        """FRP por píxel que vería el sensor en cada miembro, sin sesgo. (N, n_pix)."""
        return (obs.H @ self.x.reshape(self.n, -1).T).T

    def _update_bias(self, obs: Observation, hx: np.ndarray) -> None:
        """Kalman escalar sobre log(b) con la innovación media de píxeles fuertes."""
        if self.last_viirs_h is None:
            return
        b = np.exp(self.log_b)
        strong = obs.detected & (b * hx.mean(axis=0) > 3.0 * obs.thr_mw)
        if strong.sum() < 3:
            return
        self.log_b_var += self.cfg.bias_walk_sd ** 2
        e = 0.1 * obs.thr_mw
        log_pred = np.log(hx[:, strong] + e)                # (N, k)
        # el análisis vive en log: la referencia es la media geométrica del ensemble
        delta = np.log(obs.frp[strong] + e) - log_pred.mean(axis=0)
        stale = self.cfg.bias_drift_per_h * (obs.time_h - self.last_viirs_h)
        # el estado solo sirve de referencia en la medida en que el último VIIRS
        # no lo sorprendió: esa innovación entra como error de la referencia
        r = (obs.sensor.sigma_log ** 2 / strong.sum()
             + log_pred.mean(axis=1).var(ddof=1) + stale ** 2 + self.viirs_surprise ** 2)
        gain = self.log_b_var / (self.log_b_var + r)
        self.log_b += gain * (np.median(delta) - self.log_b)
        self.log_b_var *= 1.0 - gain

    def _update_params(self, obs: Observation, pred: np.ndarray) -> None:
        """Ajusta las tasas del modelo con el FRP total de la pasada (EnSRF escalar).

        Los miembros con tasas más altas pronostican más FRP total; esa covarianza
        permite corregir beta, x_peak y tau_res cuando el pronóstico se pasa o se
        queda corto. Una sola observación escalar evita correlaciones espurias.
        """
        cfg, thr = self.cfg, obs.thr_mw
        visible = obs.detected | (obs.clear & (pred > thr))       # (N, n_pix)
        h = np.log((pred * visible).sum(axis=1) + thr)
        y = np.log(obs.frp[obs.detected].sum() + thr)
        var = h.var(ddof=1)
        if var < 1e-8:
            return
        r = cfg.param_obs_sd ** 2
        alpha = 1.0 / (1.0 + np.sqrt(r / (var + r)))
        dh = h - h.mean()
        for name in self.params.RATES:
            p = np.log(np.asarray(getattr(self.params, name), dtype=float))
            if p.ndim == 0:
                continue
            dp = p - p.mean()
            gain = (dp @ dh) / (self.n - 1) / (var + r)
            shift = float(np.clip(gain * (y - h.mean()), -0.3, 0.3))
            new = p.mean() + shift + dp - alpha * gain * dh
            sd = new.std(ddof=1)
            if sd < cfg.param_min_sd:
                new = new.mean() + (new - new.mean()) * cfg.param_min_sd / max(sd, 1e-12)
            setattr(self.params, name, np.exp(new))

    def _birth(self, obs: Observation, pred: np.ndarray, b: float) -> int:
        """Enciende fuego en los miembros que no explican una detección."""
        xf = self.x.reshape(self.n, -1)
        ff = self.f.reshape(self.n, -1)
        prior = xf.mean(axis=0)
        H, born = obs.H, 0
        # intensidad de equilibrio de una celda con combustible completo, por miembro
        x_full = np.broadcast_to(np.asarray(self.params.x_peak, dtype=float), (self.n,)) \
            * diurnal(obs.time_h, self.params)
        for o in np.flatnonzero(obs.detected):
            lacking = np.flatnonzero(pred[:, o] < self.cfg.birth_frac * obs.frp[o])
            if lacking.size == 0:
                continue
            cells = H.indices[H.indptr[o]:H.indptr[o + 1]]
            w = H.data[H.indptr[o]:H.indptr[o + 1]]
            xy = self.cell_xy[cells]
            for j in lacking:
                # más probable donde el ensemble ya veía fuego y donde queda combustible
                q = w * (prior[cells] + 0.05 * obs.frp[o] / cells.size) * (ff[j, cells] + 0.05)
                k = self.rng.choice(cells.size, p=q / q.sum())
                blob = np.exp(-((xy - xy[k]) ** 2).sum(axis=1) / (2 * CELL_KM ** 2))
                amount = obs.frp[o] / b * np.exp(obs.sensor.sigma_log * self.rng.standard_normal())
                # reparto que conserva la energía observada (las columnas de H suman 1)
                xf[j, cells] += amount * w * blob / (w * blob).sum()
                # el combustible de las celdas recién encendidas es el que explica su
                # intensidad: sin esto el modelo las llevaría al máximo en la hora siguiente
                lit = cells[xf[j, cells] > self.params.x_off]
                implied = np.clip(xf[j, lit] / x_full[j], 0.05, 1.0)
                w_f = self.cfg.birth_fuel_weight
                ff[j, lit] = ff[j, lit] ** (1.0 - w_f) * implied ** w_f
                born += 1
        return born

    def assimilate(self, obs: Observation) -> dict:
        cfg = self.cfg
        is_modis = obs.sensor.family == "MODIS"
        hx = self._hx(obs)
        if is_modis and cfg.estimate_bias:
            self._update_bias(obs, hx)
        b = float(np.exp(self.log_b)) if is_modis else 1.0
        if not is_modis and obs.detected.any():
            e = 0.1 * obs.thr_mw
            d = obs.detected
            self.viirs_surprise = float(abs(np.mean(
                np.log(obs.frp[d] + e) - np.log(hx[:, d] + e).mean(axis=0))))

        if cfg.estimate_params and self.n_assimilated > 0:
            self._update_params(obs, b * hx)
        self.n_assimilated += 1

        born = self._birth(obs, b * hx, b)
        pred = b * self._hx(obs)

        e = 0.1 * obs.thr_mw
        det = obs.detected
        # no-detección censurada: solo informa si algún miembro supera el umbral
        non = obs.clear & ~det & (pred.max(axis=0) > obs.thr_mw)
        cap = cfg.nondet_level * obs.thr_mw
        yb = np.concatenate([np.log(pred[:, det] + e),
                             np.log(np.maximum(pred[:, non], cap) + e)], axis=1)
        yo = np.concatenate([np.log(obs.frp[det] + e), np.full(non.sum(), np.log(cap + e))])
        r_det = obs.sensor.sigma_log ** 2 + (0.1 * obs.thr_mw / obs.frp[det]) ** 2
        if is_modis:
            r_det = r_det + self.log_b_var
        r = np.concatenate([r_det, np.full(non.sum(), cfg.sigma_nondet ** 2)])
        obs_xy = np.concatenate([obs.centers_km[det], obs.centers_km[non]])

        z = np.log(self.x.reshape(self.n, -1) + cfg.eps)
        f = self.f.reshape(self.n, -1)
        cells = letkf_update([z, f], yb, yo, r, self.cell_xy, obs_xy,
                             cfg.loc_km + 0.5 * max(obs.ds_km, obs.dt_km), cfg.infl)
        x = np.exp(z) - cfg.eps
        self.x = np.where(x < X_MIN, 0.0, x).reshape(self.x.shape)
        self.f = np.clip(f, 0.0, 1.0).reshape(self.f.shape)
        if not is_modis:
            self.last_viirs_h = obs.time_h
        return {"n_det": int(det.sum()), "n_nondet": int(non.sum()),
                "born": born, "cells": cells, "bias": float(np.exp(self.log_b)),
                # lo que el pronóstico esperaba ver en los píxeles detectados, y lo visto
                "pred_sum": float(b * hx[:, det].mean(axis=0).sum()),
                **{name: float(np.exp(np.mean(np.log(getattr(self.params, name)))))
                   for name in self.params.RATES},
                "obs_sum": float(obs.frp[det].sum())}


@dataclass
class RunResult:
    label: str
    mean: np.ndarray             # (T, ny, nx) media del ensemble (o línea de base)
    spread: np.ndarray | None    # (T, ny, nx) desvío del ensemble
    total_members: np.ndarray | None   # (T, N) FRP total de cada miembro
    bias_trace: list[tuple[float, float, float]]   # (hora, b estimado, desvío de log b)
    log: list[dict]
    times: np.ndarray | None = None    # (T,) horas de cada campo guardado
    verification: list[dict] | None = None   # una entrada por pasada reservada


def verify_pass(flt: FireFilter, obs: Observation, naive: np.ndarray) -> dict:
    """Compara una pasada NO asimilada con lo que predicen el filtro y la persistencia.

    El filtro predice con la media del ensemble vista a través del H de esa pasada;
    la persistencia, con el mapa grillado de la última pasada asimilada. Solo
    cuentan los píxeles despejados y, en las sumas, el FRP por encima del umbral.
    """
    b = float(np.exp(flt.log_b)) if obs.sensor.family == "MODIS" else 1.0
    out = {"time_h": obs.time_h, "platform": obs.platform, "family": obs.sensor.family,
           "obs_total": float(np.nansum(obs.frp[obs.detected])),
           "n_detected": int(obs.detected.sum())}
    for name, field in (("filtro", flt.x.mean(axis=0)), ("persistencia", naive)):
        pred = b * (obs.H @ field.ravel()) if name == "filtro" else obs.H @ field.ravel()
        fire = obs.clear & (pred > obs.thr_mw)
        hits = int((fire & obs.detected).sum())
        out[f"{name}_total"] = float(pred[fire].sum())
        out[f"{name}_hits"] = hits
        out[f"{name}_misses"] = int(obs.detected.sum()) - hits
        out[f"{name}_false"] = int((fire & ~obs.detected).sum())
    return out


def run_passes(flt: FireFilter, passes: dict[int, list[Observation]], times: np.ndarray,
               label: str, store_every: int = 1,
               holdout: dict[int, list[Observation]] | None = None) -> RunResult:
    """Avanza el filtro por `times` asimilando passes[k] al llegar al paso k.

    Guarda media y dispersión cada `store_every` pasos (y el FRP total siempre).
    Las pasadas de `holdout` no se asimilan: solo se comparan con el pronóstico.
    """
    naive = np.zeros((flt.ny, flt.nx))
    verification = []
    T = len(times) - 1
    keep = np.arange(0, T + 1, store_every)
    mean = np.zeros((keep.size, flt.ny, flt.nx))
    spread = np.zeros_like(mean)
    totals = np.zeros((T + 1, flt.n))
    bias_trace, log = [], []
    for k in range(1, T + 1):
        flt.forecast(float(times[k] - times[k - 1]), float(times[k - 1]))
        for obs in (holdout or {}).get(k, []):
            verification.append(verify_pass(flt, obs, naive))
        for obs in passes.get(k, []):
            info = flt.assimilate(obs)
            naive = grid_naive(obs, (flt.ny, flt.nx))
            log.append({"time_h": obs.time_h, "platform": obs.platform,
                        "family": obs.sensor.family, **info})
            if obs.sensor.family == "MODIS" and flt.cfg.estimate_bias:
                bias_trace.append((obs.time_h, info["bias"], float(np.sqrt(flt.log_b_var))))
        totals[k] = flt.x.sum(axis=(1, 2))
        if k % store_every == 0:
            mean[k // store_every] = flt.x.mean(axis=0)
            spread[k // store_every] = flt.x.std(axis=0, ddof=1)
    return RunResult(label, mean, spread, totals, bias_trace, log, times[keep], verification)
