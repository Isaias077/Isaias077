"""Agregación temporal del FRP armonizado: por día, por semana y por mes.

Convierte los campos horarios del filtro en mapas de energía radiada por período
(el "mapa de focos de calor" de ese día, semana o mes) y en indicadores para ver
patrones y tendencias: día de máxima actividad de cada celda, tendencia entre
períodos y perfil horario.

Los períodos se cortan en hora solar local: un día va de medianoche a medianoche
local, una semana es la semana ISO (lunes a domingo) y un mes es el mes calendario.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

SCALES = ("dia", "semana", "mes")


def local_datetimes(utc: np.ndarray, solar_offset_h: float) -> list[datetime]:
    """Hora solar local de cada instante UTC (cadenas ISO o datetime64)."""
    shift = timedelta(hours=float(solar_offset_h))
    return [datetime.fromisoformat(str(u)[:16]) + shift for u in np.asarray(utc)]


def period_key(when: datetime, scale: str) -> str:
    if scale == "dia":
        return when.strftime("%Y-%m-%d")
    if scale == "semana":
        iso = when.isocalendar()
        return f"{iso[0]}-S{iso[1]:02d}"
    if scale == "mes":
        return when.strftime("%Y-%m")
    raise ValueError(f"escala desconocida: {scale}. Opciones: {', '.join(SCALES)}")


@dataclass
class Periods:
    scale: str
    labels: list[str]            # un rótulo por período, en orden
    energy_gwh: np.ndarray       # (P, ny, nx) energía radiada de cada celda en el período
    hours: np.ndarray            # (P,) horas de datos que cubre cada período
    full_hours: np.ndarray       # (P,) horas que tendría el período completo

    @property
    def complete(self) -> np.ndarray:
        """Períodos cubiertos casi por completo; los parciales no se comparan entre sí."""
        return self.hours >= 0.9 * self.full_hours

    @property
    def total_gwh(self) -> np.ndarray:
        return self.energy_gwh.sum(axis=(1, 2))

    @property
    def burning_cells(self) -> np.ndarray:
        """Celdas con actividad apreciable en cada período (más de 1 MW h)."""
        return (self.energy_gwh > 1e-3).sum(axis=(1, 2))


def _full_hours(label: str, scale: str) -> float:
    if scale == "dia":
        return 24.0
    if scale == "semana":
        return 168.0
    year, month = (int(v) for v in label.split("-"))
    nxt = datetime(year + month // 12, month % 12 + 1, 1)
    return (nxt - datetime(year, month, 1)).total_seconds() / 3600.0


def aggregate(frp_mw: np.ndarray, when: list[datetime], scale: str) -> Periods:
    """Suma los campos (T, ny, nx) en MW por período. `when` es la hora local de cada uno."""
    step_h = float(np.median(np.diff([w.timestamp() for w in when]))) / 3600.0
    keys = [period_key(w, scale) for w in when]
    labels = sorted(set(keys))
    index = np.array([labels.index(k) for k in keys])
    energy = np.zeros((len(labels),) + frp_mw.shape[1:])
    np.add.at(energy, index, frp_mw * step_h / 1000.0)
    hours = np.bincount(index, minlength=len(labels)) * step_h
    return Periods(scale, labels, energy, hours,
                   np.array([_full_hours(lab, scale) for lab in labels]))


def peak_period(p: Periods) -> np.ndarray:
    """Índice del período de máxima actividad de cada celda; -1 donde no hubo fuego."""
    peak = p.energy_gwh.argmax(axis=0)
    return np.where(p.energy_gwh.sum(axis=0) > 1e-3, peak, -1)


def trend(p: Periods) -> np.ndarray:
    """Pendiente de la energía por celda entre períodos completos [GW h por período].

    Positiva donde la actividad va en aumento. Con menos de tres períodos completos
    no hay tendencia que medir y devuelve ceros.
    """
    e = p.energy_gwh[p.complete]
    if e.shape[0] < 3:
        return np.zeros(p.energy_gwh.shape[1:])
    t = np.arange(e.shape[0]) - (e.shape[0] - 1) / 2.0
    return np.tensordot(t, e, axes=(0, 0)) / (t ** 2).sum()


def hourly_profile(frp_mw: np.ndarray, when: list[datetime]) -> np.ndarray:
    """FRP total medio del dominio para cada hora solar local (24 valores)."""
    total = frp_mw.sum(axis=(1, 2))
    hour = np.array([w.hour for w in when])
    return np.array([total[hour == h].mean() if (hour == h).any() else np.nan for h in range(24)])


def detection_density(lat: np.ndarray, lon: np.ndarray, when: list[datetime], scale: str,
                      labels: list[str], bbox: tuple[float, float, float, float],
                      shape: tuple[int, int]) -> np.ndarray:
    """Cantidad de detecciones crudas por celda y período, en la misma grilla. (P, ny, nx)"""
    ny, nx = shape
    iy = np.clip(((lat - bbox[1]) / (bbox[3] - bbox[1]) * ny).astype(int), 0, ny - 1)
    ix = np.clip(((lon - bbox[0]) / (bbox[2] - bbox[0]) * nx).astype(int), 0, nx - 1)
    out = np.zeros((len(labels), ny, nx))
    for k, w in enumerate(when):
        key = period_key(w, scale)
        if key in labels:
            out[labels.index(key), iy[k], ix[k]] += 1
    return out
