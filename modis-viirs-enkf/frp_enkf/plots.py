"""Figuras del experimento (matplotlib, PNG estáticos)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .model import CELL_KM
from .osse import RunResult, Scenario, Truth
from .sensors import grid_naive

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_MODIS, C_VIIRS, C_FILTER, C_BASE = "#2a78d6", "#eb6834", "#1baf7a", "#9c9b95"
FIRE = LinearSegmentedColormap.from_list("fire", [SURFACE, "#f6b89c", "#eb6834", "#7a2c0e"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def _day_axis(ax, truth: Truth) -> None:
    t0, t1 = truth.times[0], truth.times[-1]
    ticks = np.arange(np.ceil(t0 / 12) * 12, t1 + 0.1, 12)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"día {int(t // 24) + 1}\n{int(t % 24):02d}:00" for t in ticks])
    ax.set_xlim(t0, t1)
    ax.set_ylim(bottom=0)


def fig_problem(truth: Truth, path: str) -> None:
    """FRP total real contra lo que suma cada pasada cruda."""
    tot = truth.x.sum(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(truth.times, tot, color=INK, lw=2, label="Verdad (grilla de 375 m)")
    for fam, color, marker in (("MODIS", C_MODIS, "o"), ("VIIRS", C_VIIRS, "s")):
        pts = [(o.time_h, np.nansum(o.frp)) for ps in truth.passes.values() for o in ps
               if o.sensor.family == fam]
        ax.scatter(*zip(*pts), s=52, color=color, marker=marker, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=f"Suma de detecciones {fam}")
    ax.set_ylabel("FRP total del incendio [MW]")
    ax.set_title("El problema: cada sensor cuenta una historia distinta del mismo incendio")
    ax.legend(loc="upper left")
    _day_axis(ax, truth)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_total(truth: Truth, res: RunResult, naive: RunResult, path: str) -> None:
    """FRP total: verdad, filtro con su banda de incertidumbre y línea de base."""
    tot = truth.x.sum(axis=(1, 2))
    lo, hi = np.percentile(res.total_members, [5, 95], axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(truth.times, lo, hi, color=C_FILTER, alpha=0.25, lw=0,
                    label="Filtro: 5–95 % del ensemble")
    ax.plot(truth.times, naive.mean.sum(axis=(1, 2)), color=C_BASE, lw=1.5,
            drawstyle="steps-post", label="Línea de base: última pasada grillada")
    ax.plot(truth.times, tot, color=INK, lw=2, label="Verdad")
    ax.plot(truth.times, res.mean.sum(axis=(1, 2)), color=C_FILTER, lw=2,
            label="Filtro: media del ensemble")
    for k, ps in truth.passes.items():
        for o in ps:
            ax.plot([o.time_h] * 2, [0, -0.03 * tot.max()], lw=2, clip_on=False,
                    color=C_MODIS if o.sensor.family == "MODIS" else C_VIIRS)
    ax.set_ylabel("FRP total del incendio [MW]")
    ax.set_title("El filtro reconstruye el FRP continuo entre pasadas "
                 "(marcas: azul MODIS, naranja VIIRS)")
    ax.legend(loc="upper left")
    _day_axis(ax, truth)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pick_showcase_pass(truth: Truth) -> tuple[int, object]:
    """La pasada MODIS con el píxel más grande que todavía detecta fuego."""
    cands = [(o.area_km2, k, o) for k, ps in truth.passes.items() for o in ps
             if o.sensor.family == "MODIS" and o.detected.sum() >= 5 and o.clear.all()]
    _, k, o = max(cands, key=lambda c: c[0])
    return k, o


def fig_maps(sc: Scenario, truth: Truth, res: RunResult, path: str) -> None:
    """Mapas en una pasada MODIS oblicua: verdad, dato crudo, análisis y dispersión."""
    k, obs = pick_showcase_pass(truth)
    raw = grid_naive(obs, sc.shape)
    panels = [
        ("Verdad (375 m)", truth.x[k]),
        (f"{obs.sensor.name} crudo: píxel de {obs.ds_km:.1f} × {obs.dt_km:.1f} km", raw),
        ("Filtro: media del ensemble", res.mean[k]),
        ("Filtro: dispersión del ensemble", res.spread[k]),
    ]
    ys, xs = np.nonzero((truth.x[k] > 0.5) | (raw > 0.05))
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, sc.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, sc.shape[1])
    vmax = float(np.percentile(truth.x[k][truth.x[k] > 0.5], 98))
    ext = [x0 * CELL_KM, x1 * CELL_KM, y0 * CELL_KM, y1 * CELL_KM]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (title, fld) in zip(axes, panels):
        im = ax.imshow(fld[y0:y1, x0:x1], origin="lower", extent=ext, cmap=FIRE,
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("km")
        ax.grid(False)
    axes[0].set_ylabel("km")
    fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01, label="FRP por celda de 375 m [MW]")
    fig.suptitle(f"Pasada de {obs.platform} a las {int(obs.time_h % 24):02d}:"
                 f"{int(obs.time_h % 1 * 60):02d} del día {int(obs.time_h // 24) + 1}, "
                 f"ángulo de barrido {abs(obs.theta_deg):.0f}°", x=0.01, ha="left",
                 fontweight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_scores(scores: dict[str, dict], res: RunResult, true_bias: float, path: str) -> None:
    """Error por configuración y convergencia del sesgo MODIS/VIIRS estimado."""
    groups = ["MODIS + VIIRS", "Solo VIIRS", "Solo MODIS"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(groups))
    for off, key, color, lab in ((-0.19, "Filtro", C_FILTER, "Filtro"),
                                 (0.19, "Base", C_BASE, "Línea de base")):
        vals = [scores[f"{key} · {g}"]["rmse_375m"] for g in groups]
        a.barh(y + off, vals, height=0.34, color=color, label=lab)
        for yi, v in zip(y + off, vals):
            a.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    a.set_yticks(y)
    a.set_yticklabels(groups)
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    a.set_xlabel("RMSE por celda de 375 m [MW]")
    a.set_title("Error contra la verdad (menor es mejor)")
    a.legend(loc="upper right")

    t, bh, sd = (np.array(v) for v in zip(*res.bias_trace))
    b.axhline(true_bias, color=INK, lw=1.5, ls="--")
    b.text(0.98, 0.95, f"línea punteada: valor real = {true_bias:.2f}", ha="right", va="top",
           fontsize=9, color=INK2, transform=b.transAxes)
    b.fill_between(t, bh * np.exp(-sd), bh * np.exp(sd), color=C_MODIS, alpha=0.22, lw=0)
    b.plot(t, bh, color=C_MODIS, lw=2, marker="o", ms=5, mec=SURFACE)
    b.set_xlabel("hora desde la medianoche del día 1")
    b.set_ylabel("b = FRP MODIS / FRP VIIRS")
    b.set_title("Sesgo MODIS estimado en línea (±1σ)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
