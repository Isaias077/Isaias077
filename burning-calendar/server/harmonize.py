"""Armonización de conteos mensuales de hotspots MODIS y VIIRS.

El resultado es una sola serie mensual en "hotspots equivalentes VIIRS" (S-NPP,
375 m) desde 2000 hasta hoy, para que el sensor de origen deje de importar:

1. Terra solo -> Terra + Aqua. Antes de julio de 2002 solo volaba Terra. Sus
   conteos se llevan a la escala de los dos satélites con el cociente
   (Terra + Aqua) / Terra de cada mes del año, medido cuando volaban ambos.
2. MODIS -> VIIRS. En los años en que conviven (2012 en adelante) se ajusta
   log(1 + VIIRS) = a + b * log(1 + MODIS) para el área elegida. La relación no
   es un factor fijo: VIIRS ve fuegos chicos que MODIS pierde, así que el
   cociente cambia con el nivel de actividad.
3. Desde 2012 la serie usa VIIRS observado; antes, MODIS convertido con ese
   ajuste, con una banda de incertidumbre tomada de sus residuos.

La calidad se mide dejando un año afuera: se ajusta con los demás años y se
predice el VIIRS de ese año solo con MODIS.
"""
from __future__ import annotations

import numpy as np

AQUA_START = (2002 - 2000) * 12 + 6      # julio de 2002, en meses desde enero de 2000
VIIRS_START = (2012 - 2000) * 12 + 1     # febrero de 2012: primer mes completo de S-NPP
MIN_OVERLAP_MONTHS = 18
Z90 = 1.645


def terra_to_modis(month: np.ndarray, terra: np.ndarray, aqua: np.ndarray) -> np.ndarray:
    """Conteo MODIS de dos satélites; en la era de Terra solo, Terra escalado."""
    both = month >= AQUA_START
    total = (terra + aqua).astype(float)
    out = total.copy()
    moy = month % 12
    overall = total[both].sum() / max(terra[both].sum(), 1.0)
    for m in range(12):
        sel = both & (moy == m)
        k = total[sel].sum() / terra[sel].sum() if terra[sel].sum() >= 50 else overall
        early = ~both & (moy == m)
        out[early] = terra[early] * k
    return out


def fit_loglog(modis: np.ndarray, viirs: np.ndarray) -> tuple[float, float, float]:
    """Ajuste log(1+V) = a + b log(1+M). Devuelve (a, b, desvío de los residuos)."""
    x, y = np.log1p(modis), np.log1p(viirs)
    if np.ptp(x) < 1e-9:
        return float(y.mean() - x.mean()), 1.0, float(y.std())
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    return float(a), float(b), float(resid.std(ddof=2)) if x.size > 2 else 0.0


def predict(modis: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.maximum(np.expm1(a + b * np.log1p(modis)), 0.0)


def _factor(pred: np.ndarray, obs: np.ndarray) -> float:
    """Error típico como factor: exp(mediana de |log((pred+1)/(obs+1))|)."""
    return float(np.exp(np.median(np.abs(np.log((pred + 1.0) / (obs + 1.0))))))


def cross_validate(month: np.ndarray, modis: np.ndarray, viirs: np.ndarray) -> dict:
    """Deja un año afuera, ajusta con el resto y predice ese año solo con MODIS."""
    year = month // 12
    pred = np.zeros_like(modis, dtype=float)
    for y in np.unique(year):
        test = year == y
        a, b, _ = fit_loglog(modis[~test], viirs[~test])
        pred[test] = predict(modis[test], a, b)
    years = np.unique(year)
    ann_pred = np.array([pred[year == y].sum() for y in years])
    ann_obs = np.array([viirs[year == y].sum() for y in years])
    return {
        "factor_mensual": _factor(pred, viirs),
        "factor_anual": _factor(ann_pred, ann_obs),
        "factor_mensual_sin_armonizar": _factor(modis, viirs),
        "factor_anual_sin_armonizar": _factor(
            np.array([modis[year == y].sum() for y in years]), ann_obs),
        "anios": int(years.size),
    }


def apply(month: np.ndarray, terra: np.ndarray, aqua: np.ndarray, viirs: np.ndarray,
          viirs_end: int, a: float, b: float, sigma: float) -> dict:
    """Serie armonizada con un ajuste ya hecho: VIIRS hasta viirs_end, MODIS convertido fuera."""
    modis = terra_to_modis(month, terra, aqua)
    from_viirs = (month >= VIIRS_START) & (month <= viirs_end)
    est = predict(modis, a, b)
    lo = np.maximum(np.expm1(np.log1p(est) - Z90 * sigma), 0.0)
    hi = np.expm1(np.log1p(est) + Z90 * sigma)
    return {"modis": modis, "harmonized": np.where(from_viirs, viirs, est),
            "low": np.where(from_viirs, viirs, lo), "high": np.where(from_viirs, viirs, hi),
            "from_viirs": from_viirs}


def harmonize(month: np.ndarray, terra: np.ndarray, aqua: np.ndarray, viirs: np.ndarray,
              viirs_end: int, fallback: tuple[float, float, float] | None = None) -> dict:
    """Serie mensual armonizada.

    month: meses desde enero de 2000, consecutivos. viirs_end: último mes con VIIRS.
    fallback: ajuste (a, b, sigma) de un área mayor, para áreas con pocos datos.
    """
    modis = terra_to_modis(month, terra, aqua)
    overlap = (month >= VIIRS_START) & (month <= viirs_end)
    active = overlap & ((modis > 0) | (viirs > 0))
    local = int(active.sum()) >= MIN_OVERLAP_MONTHS
    if local:
        a, b, sigma = fit_loglog(modis[overlap], viirs[overlap])
        cv = cross_validate(month[overlap], modis[overlap], viirs[overlap])
        origin = "area"
    elif fallback is not None:
        (a, b, sigma), cv, origin = fallback, None, "pais"
    else:
        a, b, sigma, cv, origin = 0.0, 1.0, 0.0, None, "ninguno"

    return {**apply(month, terra, aqua, viirs, viirs_end, a, b, sigma),
            "fit": {"a": a, "b": b, "sigma": sigma, "origen": origin,
                    "meses_solapados": int(active.sum())},
            "cv": cv}


def calendar_stats(month: np.ndarray, value: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Climatología por mes del año y anomalía de cada mes frente a los demás años.

    Los meses con valid=False (por ejemplo, el mes en curso, incompleto) no entran en
    la climatología ni en la referencia de los demás; su anomalía se calcula igual,
    pero conviene reemplazarla por una comparación de los mismos días.
    """
    moy = month % 12
    valid = np.ones(value.size, dtype=bool) if valid is None else valid
    median = np.zeros(12)
    p25, p75 = np.zeros(12), np.zeros(12)
    anomaly = np.zeros(value.size)       # cociente frente a la mediana de los otros años
    rank = np.zeros(value.size)          # percentil dentro de su mes del año
    for m in range(12):
        idx = np.flatnonzero(moy == m)
        ok = valid[idx]
        v = value[idx]
        if not ok.any():
            continue
        median[m], p25[m], p75[m] = np.median(v[ok]), *np.percentile(v[ok], [25, 75])
        for j, i in enumerate(idx):
            others = v[ok & (np.arange(idx.size) != j)]
            ref = np.median(others) if others.size else v[j]
            anomaly[i] = (v[j] + 1.0) / (ref + 1.0)
            rank[i] = 100.0 * (others < v[j]).mean() if others.size else 50.0
    return {"median": median, "p25": p25, "p75": p75, "anomaly": anomaly, "rank": rank}


def same_days_comparison(month: np.ndarray, value: np.ndarray, target: int) -> dict:
    """Compara el mes `target` con el mismo mes de los demás años (mismos días).

    `value` debe venir contado solo hasta el mismo día del mes en todos los años.
    """
    sel = (month % 12 == target % 12) & (month != target)
    ref = value[sel]
    cur = float(value[month == target][0])
    med = float(np.median(ref)) if ref.size else cur
    return {"actual": cur, "mediana": med, "anomalia": (cur + 1.0) / (med + 1.0),
            "percentil": float(100.0 * (ref < cur).mean()) if ref.size else 50.0,
            "anios_de_referencia": int(ref.size)}
