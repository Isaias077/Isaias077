#!/usr/bin/env python3
"""Gráficas de estado: cómo está quedando el filtro, en sintético y con datos reales.

    python make_status_figures.py            # ~3 minutos; escribe figures/estado/
    python make_status_figures.py --replot   # redibuja con los datos ya calculados

Corre el experimento sintético con cuatro semillas y lee los resultados de
Queensland ya generados por run_firms.py (con y sin satélite reservado).
"""
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from frp_enkf.osse import Scenario, run_filter, run_naive, score, simulate_truth
from frp_enkf.plots import C_BASE, C_FILTER, C_MODIS, C_VIIRS, GRID, INK, INK2, SURFACE
from frp_enkf.sensors import MODIS

SEEDS = (7, 1, 2, 3)
OUT = "figures/estado"
PEAK_OLD_MODEL_MW = 7614.0   # pico de Queensland con el modelo de crecimiento exponencial


def run_seeds() -> list[dict]:
    rows = []
    for seed in SEEDS:
        sc = Scenario(seed=seed)
        truth = simulate_truth(sc)
        t0 = truth.times[min(truth.passes)] + 6.0
        fam = ("MODIS", "VIIRS")
        res = run_filter(sc, truth, fam, "filtro")
        rows.append({
            "seed": seed,
            "wind_start": (float(sc.model.wind_x), float(sc.model.wind_y)),
            "wind_true": (float(sc.truth.wind_x), float(sc.truth.wind_y)),
            "filtro": score(sc, truth, res, t0),
            "base": score(sc, truth, run_naive(sc, truth, fam, "base"), t0),
            "bias": np.array(res.bias_trace),
            "wind": np.array([(r["time_h"], r["wind_x"], r["wind_y"]) for r in res.log]),
        })
        print(f"semilla {seed} lista")
    return rows


def _bars(ax, rows, key, title, unit) -> None:
    y = np.arange(len(rows))
    for off, name, color, lab in ((-0.19, "filtro", C_FILTER, "Filtro"),
                                  (0.19, "base", C_BASE, "Línea de base")):
        vals = [r[name][key] for r in rows]
        ax.barh(y + off, vals, height=0.34, color=color, label=lab)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.012 * ax.get_xlim()[1], yi, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                    va="center", fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"semilla {r['seed']}" for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(unit)
    ax.set_title(title)


def fig_synthetic(rows, path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.set_xlim(0, 1.15 * max(r["base"]["rmse_375m"] for r in rows))
    b.set_xlim(0, 1.15 * max(r["base"]["total_mae_pct"] for r in rows))
    _bars(a, rows, "rmse_375m", "Error por celda de 375 m", "RMSE [MW] (menor es mejor)")
    _bars(b, rows, "total_mae_pct", "Error del FRP total del incendio", "error medio [%]")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Experimento sintético: filtro contra línea de base, MODIS + VIIRS",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_bias_wind(rows, path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    a.axhline(MODIS.bias, color=INK, lw=1.5, ls="--")
    for r in rows:
        t, bh, _ = r["bias"].T
        a.step(t, bh, where="post", color=C_MODIS, lw=1.8, alpha=0.85)
    finals = "\n".join(f"semilla {r['seed']}: {r['bias'][-1, 1]:.2f}" for r in rows)
    a.text(0.97, 0.95, f"línea punteada: valor real = {MODIS.bias:.2f}\n\nvalores finales\n{finals}",
           fontsize=9, color=INK2, ha="right", va="top", transform=a.transAxes)
    a.set_xlabel("hora desde la medianoche del día 1")
    a.set_ylabel("b = FRP MODIS / FRP VIIRS")
    a.set_title("Sesgo de MODIS: solo cambia con un VIIRS a menos de 45 min")

    start, true = rows[0]["wind_start"], rows[0]["wind_true"]
    for r in rows:
        w = r["wind"]
        b.plot(np.r_[start[0], w[:, 1]], np.r_[start[1], w[:, 2]], color=C_VIIRS, lw=1.5,
               alpha=0.85)
        b.scatter(w[-1, 1], w[-1, 2], s=46, color=C_VIIRS, edgecolor=SURFACE, linewidth=1.5,
                  zorder=3)
        b.annotate(f"semilla {r['seed']}", (w[-1, 1], w[-1, 2]), xytext=(6, 6),
                   textcoords="offset points", fontsize=9, color=INK2)
    b.scatter(*start, s=70, marker="s", color=C_BASE, zorder=4)
    b.annotate("supuesto inicial", start, xytext=(-8, 8), textcoords="offset points",
               fontsize=9, ha="right")
    b.scatter(*true, s=130, marker="*", color=INK, zorder=4)
    b.annotate("viento real", true, xytext=(8, -12), textcoords="offset points", fontsize=9)
    b.set_xlabel("componente x (hacia el este)")
    b.set_ylabel("componente y (hacia el norte)")
    b.set_aspect("equal", adjustable="datalim")
    b.set_title("Viento efectivo: recorrido de la estimación")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_real(path) -> None:
    with open("figures/queensland/resumen.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    with open("figures/queensland_validacion/validacion.csv", encoding="utf-8") as fh:
        val = list(csv.DictReader(fh))
    obs = np.array([float(v["obs_total"]) for v in val])
    filt = np.array([float(v["filtro_total"]) for v in val])
    pers = np.array([float(v["persistencia_total"]) for v in val])

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    names = ["Mayor suma observada\nen una pasada", "Filtro: modelo actual",
             "Filtro: modelo anterior"]
    vals = [summary["suma_maxima_de_una_pasada_mw"], summary["frp_total_maximo_mw"],
            PEAK_OLD_MODEL_MW]
    a.barh(names, vals, color=[INK2, C_FILTER, C_BASE], height=0.55)
    for i, v in enumerate(vals):
        a.text(v + 100, i, f"{v:,.0f} MW".replace(",", " "), va="center", fontsize=10)
    a.set_xlim(0, 1.25 * max(vals))
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    a.set_xlabel("FRP total máximo en el dominio [MW]")
    a.set_title("Queensland: la sobreestimación bajó a la mitad")

    floor = 1.0
    lim = (5, 1.6 * max(obs.max(), filt.max(), pers.max()))
    b.plot(lim, lim, color=INK, lw=1.2, ls="--")
    b.scatter(obs + floor, pers + floor, s=46, marker="s", color=C_BASE, edgecolor=SURFACE,
              linewidth=1.2, label="Persistencia (última pasada)", zorder=3)
    b.scatter(obs + floor, filt + floor, s=52, color=C_FILTER, edgecolor=SURFACE,
              linewidth=1.2, label="Filtro", zorder=4)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(*lim)
    b.set_ylim(0.8, lim[1])
    b.text(0.97, 0.05, "línea punteada: pronóstico perfecto", ha="right", fontsize=9,
           color=INK2, transform=b.transAxes)
    b.set_xlabel("FRP total observado por NOAA-21 [MW]")
    b.set_ylabel("FRP total pronosticado [MW]")
    b.set_title(f"Validación con {len(val)} pasadas de NOAA-21 no asimiladas")
    b.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "datos.json")
    if "--replot" in sys.argv and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            rows = json.load(fh)
        for r in rows:
            r["bias"], r["wind"] = np.array(r["bias"]), np.array(r["wind"])
    else:
        rows = run_seeds()
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump([{**r, "bias": r["bias"].tolist(), "wind": r["wind"].tolist()}
                       for r in rows], fh)
    fig_synthetic(rows, os.path.join(OUT, "1_sintetico_semillas.png"))
    fig_bias_wind(rows, os.path.join(OUT, "2_sesgo_y_viento.png"))
    fig_real(os.path.join(OUT, "3_queensland.png"))
    print(f"Figuras en {OUT}/")


if __name__ == "__main__":
    main()
