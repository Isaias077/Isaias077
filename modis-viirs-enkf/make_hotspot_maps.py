#!/usr/bin/env python3
"""Mapas de focos de calor por día, semana y mes, para ver patrones y tendencias.

    python make_hotspot_maps.py --run-dir figures/queensland \\
        --csv-dir data/queensland_2026-11 --out figures/focos

Lee los campos horarios que dejó run_firms.py (frp_armonizado.npz) y las detecciones
crudas de FIRMS, y los agrega en hora solar local. Para tener semanas y meses
completos hay que correr antes run_firms.py sobre un período más largo.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

from frp_enkf import aggregate as ag
from frp_enkf import firms
from frp_enkf.osse import block_mean
from frp_enkf.plots import C_FILTER, C_MODIS, C_VIIRS, FIRE, GRID, INK, INK2, SURFACE

NAMES = {"dia": "día", "semana": "semana", "mes": "mes"}
DIVERGING = LinearSegmentedColormap.from_list("tend", ["#184f95", "#2a78d6", "#e4e3df",
                                                       "#eb6834", "#7a2c0e"])
ORDER = LinearSegmentedColormap.from_list("orden", ["#cde2fb", "#2a78d6", "#0d366b"])
MAX_PANELS = 8


def _label(p: ag.Periods, k: int) -> str:
    lab = p.labels[k][5:] if p.scale == "dia" else p.labels[k]
    if p.complete[k]:
        return lab
    full = "24 h" if p.scale == "dia" else f"{p.full_hours[k] / 24:.0f} días"
    have = f"{p.hours[k]:.0f} h" if p.hours[k] < 48 else f"{p.hours[k] / 24:.0f} días"
    return f"{lab}\n(parcial: {have} de {full})"


def fig_periods(p: ag.Periods, raw: dict[str, np.ndarray], bbox, path: str) -> None:
    """Una columna por período: energía armonizada y detecciones crudas de cada sensor."""
    ks = list(range(len(p.labels)))[-MAX_PANELS:]
    rows = [("Armonizado\n[GW h por celda de 375 m]", p.energy_gwh, FIRE, 1)]
    rows += [(f"{fam} crudo\n[detecciones por 1.5 km]", block_mean(raw[fam], 4) * 16, FIRE, 4)
             for fam in ("VIIRS", "MODIS") if fam in raw]
    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
    fig, axes = plt.subplots(len(rows), len(ks), figsize=(2.3 * len(ks) + 1.6, 2.5 * len(rows) + 0.7),
                             squeeze=False, layout="constrained")
    for i, (title, data, cmap, _) in enumerate(rows):
        vmax = max(float(np.percentile(data[ks], 99.7)), 1e-6)
        for j, k in enumerate(ks):
            ax = axes[i, j]
            im = ax.imshow(data[k], origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=vmax,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_color(GRID)
            if i == 0:
                ax.set_title(_label(p, k), fontsize=9, loc="center")
        axes[i, 0].set_ylabel(title, fontsize=9)
        fig.colorbar(im, ax=axes[i, :], shrink=0.85, pad=0.01)
    fig.suptitle(f"Focos de calor por {NAMES[p.scale]}: el filtro frente a lo que ve cada sensor",
                 x=0.01, ha="left", fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_patterns(daily: ag.Periods, bbox, path: str) -> None:
    """Acumulado, día de máxima actividad de cada celda y tendencia entre días."""
    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.8), layout="constrained")
    total = daily.energy_gwh.sum(axis=0)
    im = a.imshow(total, origin="lower", extent=ext, cmap=FIRE, vmin=0, aspect="auto",
                  vmax=float(np.percentile(total, 99.7)), interpolation="nearest")
    a.set_title("Dónde ardió: energía acumulada")
    fig.colorbar(im, ax=a, label="GW h por celda")

    peak = ag.peak_period(daily).astype(float)
    peak[peak < 0] = np.nan
    n = len(daily.labels)
    cmap = ListedColormap(ORDER(np.linspace(0, 1, n)))
    im = b.imshow(peak, origin="lower", extent=ext, cmap=cmap, aspect="auto",
                  norm=BoundaryNorm(np.arange(-0.5, n), n), interpolation="nearest")
    cb = fig.colorbar(im, ax=b, ticks=np.arange(n))
    cb.ax.set_yticklabels([lab[5:] for lab in daily.labels])
    b.set_title("Cuándo ardió más: día de máxima actividad")

    slope = ag.trend(daily) * 1000.0
    lim = max(float(np.percentile(np.abs(slope), 99.5)), 1e-6)
    im = c.imshow(slope, origin="lower", extent=ext, cmap=DIVERGING, vmin=-lim, vmax=lim,
                  aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=c, label="MW h por día (naranja: en aumento; azul: en baja)")
    n_ok = int(daily.complete.sum())
    c.set_title(f"Hacia dónde va: tendencia en {n_ok} días completos")
    for ax in (a, b, c):
        ax.grid(False)
        ax.set_xlabel("longitud")
    a.set_ylabel("latitud")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_trends(daily: ag.Periods, counts: dict[str, np.ndarray], profile: np.ndarray,
               path: str) -> None:
    """Evolución día a día de la energía y de las detecciones, y perfil horario."""
    x = np.arange(len(daily.labels))
    ticks = [lab[5:] for lab in daily.labels]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.2), layout="constrained")
    a.bar(x[daily.complete], daily.total_gwh[daily.complete], color=C_FILTER, width=0.7,
          label="día completo")
    a.bar(x[~daily.complete], daily.total_gwh[~daily.complete], color=C_FILTER, width=0.7,
          alpha=0.35, label="día parcial")
    for xi, v in zip(x, daily.total_gwh):
        a.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
    a.set_ylabel("energía radiada [GW h]")
    a.set_title("Energía armonizada por día")
    a.legend(loc="upper right")
    a.grid(axis="x", visible=False)

    for fam, color, marker in (("VIIRS", C_VIIRS, "s"), ("MODIS", C_MODIS, "o")):
        if fam in counts:
            b.plot(x, counts[fam], color=color, lw=2, marker=marker, ms=6, mec=SURFACE, label=fam)
            b.text(x[-1] + 0.15, counts[fam][-1], fam, va="center", fontsize=9, color=INK2)
    b.set_ylabel("detecciones crudas")
    b.set_ylim(bottom=0)
    b.set_xlim(-0.5, len(x) + 0.4)
    b.set_title("Detecciones por día de cada sensor")
    b.legend(loc="upper right")
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels(ticks)
        ax.set_xlabel("día (hora solar local)")

    c.plot(np.arange(24), profile, color=C_FILTER, lw=2)
    c.fill_between(np.arange(24), 0, profile, color=C_FILTER, alpha=0.2, lw=0)
    c.set_xticks(np.arange(0, 24, 3))
    c.set_xlim(0, 23)
    c.set_ylim(bottom=0)
    c.set_xlabel("hora solar local")
    c.set_ylabel("FRP total medio [MW]")
    c.set_title("Patrón horario del incendio")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default="figures/queensland",
                    help="carpeta con frp_armonizado.npz, salida de run_firms.py")
    ap.add_argument("--csv-dir", default="data/queensland_2026-11",
                    help="carpeta con los <FUENTE>.csv de FIRMS del mismo período")
    ap.add_argument("--out", default="figures/focos")
    args = ap.parse_args()

    run = np.load(os.path.join(args.run_dir, "frp_armonizado.npz"))
    frp, bbox = run["frp_mw"].astype(float), tuple(run["bbox"])
    offset = 0.5 * (bbox[0] + bbox[2]) / 15.0
    when = ag.local_datetimes(run["utc"], offset)

    dets = {}
    for source, (sensor, _, _) in firms.SOURCES.items():
        path = os.path.join(args.csv_dir, f"{source}.csv")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                d = firms.parse_csv(fh.read(), source, bbox)
            dets.setdefault(sensor.family, []).append(d)

    os.makedirs(args.out, exist_ok=True)
    table = []
    periods = {}
    for scale in ag.SCALES:
        p = periods[scale] = ag.aggregate(frp, when, scale)
        raw, counts = {}, {}
        for fam, ds in dets.items():
            lat = np.concatenate([d.lat for d in ds])
            lon = np.concatenate([d.lon for d in ds])
            t = ag.local_datetimes(np.concatenate([d.time for d in ds]).astype(str), offset)
            raw[fam] = ag.detection_density(lat, lon, t, scale, p.labels, bbox, frp.shape[1:])
            counts[fam] = raw[fam].sum(axis=(1, 2))
        fig_periods(p, raw, bbox, os.path.join(args.out, f"focos_por_{scale}.png"))
        for k, lab in enumerate(p.labels):
            table.append({"escala": scale, "periodo": lab, "completo": bool(p.complete[k]),
                          "horas_con_datos": round(float(p.hours[k]), 1),
                          "energia_gwh": round(float(p.total_gwh[k]), 2),
                          "celdas_con_fuego": int(p.burning_cells[k]),
                          **{f"detecciones_{f.lower()}": int(c[k]) for f, c in counts.items()}})
        if scale == "dia":
            fig_patterns(p, bbox, os.path.join(args.out, "patrones.png"))
            fig_trends(p, counts, ag.hourly_profile(frp, when),
                       os.path.join(args.out, "tendencias.png"))
        n_ok = int(p.complete.sum())
        print(f"{NAMES[scale]:>7}: {len(p.labels)} períodos, {n_ok} completos")

    with open(os.path.join(args.out, "resumen_por_escala.csv"), "w", encoding="utf-8",
              newline="") as fh:
        first = ["escala", "periodo", "completo", "horas_con_datos", "energia_gwh",
                 "celdas_con_fuego"]
        cols = first + sorted({c for row in table for c in row} - set(first))
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print(f"Figuras y tabla en {args.out}/")


if __name__ == "__main__":
    main()
