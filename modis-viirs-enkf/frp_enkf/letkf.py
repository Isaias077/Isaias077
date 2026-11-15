"""Local Ensemble Transform Kalman Filter (Hunt et al., 2007) con localización en R."""
from __future__ import annotations

import numpy as np


def gaspari_cohn(dist: np.ndarray, c: np.ndarray | float) -> np.ndarray:
    """Función de localización de Gaspari-Cohn: 1 en 0 y soporte compacto en 2c."""
    r = np.abs(dist) / c
    out = np.zeros_like(r, dtype=float)
    a = r <= 1.0
    b = (r > 1.0) & (r < 2.0)
    ra, rb = r[a], r[b]
    out[a] = -0.25 * ra**5 + 0.5 * ra**4 + 0.625 * ra**3 - (5 / 3) * ra**2 + 1.0
    out[b] = rb**5 / 12 - 0.5 * rb**4 + 0.625 * rb**3 + (5 / 3) * rb**2 \
        - 5 * rb + 4 - (2 / 3) / rb
    return np.clip(out, 0.0, 1.0)


def letkf_update(fields: list[np.ndarray], yb: np.ndarray, yo: np.ndarray,
                 r_var: np.ndarray, cell_xy: np.ndarray, obs_xy: np.ndarray,
                 obs_c: np.ndarray | float, infl: float = 1.0) -> int:
    """Actualiza en el lugar los campos del ensemble.

    fields  : lista de arreglos (N, n_celdas); a todos se les aplica la misma
              transformación local (así el combustible sigue al fuego). Solo se
              analizan las celdas donde el primer campo tiene dispersión.
    yb      : (N, m) observaciones predichas por cada miembro.
    yo      : (m,) observaciones;  r_var: (m,) varianza del error de observación.
    cell_xy : (n_celdas, 2);  obs_xy: (m, 2);  obs_c: semiancho de localización.
    Devuelve la cantidad de celdas analizadas.
    """
    n = yb.shape[0]
    if yo.size == 0:
        return 0
    ybar = yb.mean(axis=0)
    dy = yb - ybar
    innov = yo - ybar
    obs_c = np.broadcast_to(np.asarray(obs_c, dtype=float), yo.shape)

    # solo importan las observaciones con dispersión; sin ella no hay ganancia
    live = dy.std(axis=0) > 1e-9
    if not live.any():
        return 0
    dy, innov, r_var, obs_xy, obs_c = dy[:, live], innov[live], r_var[live], obs_xy[live], obs_c[live]

    reach = 2.0 * obs_c.max()
    lo, hi = obs_xy.min(axis=0) - reach, obs_xy.max(axis=0) + reach
    cand = np.flatnonzero(np.all((cell_xy >= lo) & (cell_xy <= hi), axis=1))
    # donde ningún miembro difiere en el primer campo no hay nada que corregir
    cand = cand[fields[0][:, cand].std(axis=0) > 1e-9]
    means = [fld.mean(axis=0) for fld in fields]
    eye = np.eye(n)
    done = 0
    for c in cand:
        dist = np.hypot(obs_xy[:, 0] - cell_xy[c, 0], obs_xy[:, 1] - cell_xy[c, 1])
        idx = np.flatnonzero(dist < 2.0 * obs_c)
        if idx.size == 0:
            continue
        rho = gaspari_cohn(dist[idx], obs_c[idx])
        yl = dy[:, idx]
        cw = yl * (rho / r_var[idx])
        lam, v = np.linalg.eigh(cw @ yl.T + (n - 1) / infl * eye)
        pa = (v / lam) @ v.T
        wbar = pa @ (cw @ innov[idx])
        trans = (v * np.sqrt((n - 1) / lam)) @ v.T + wbar[:, None]
        for fld, mean in zip(fields, means):
            fld[:, c] = mean[c] + (fld[:, c] - mean[c]) @ trans
        done += 1
    return done
