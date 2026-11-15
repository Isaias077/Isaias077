#!/usr/bin/env python3
"""Corre el filtro de armonización sobre detecciones reales de NASA FIRMS.

    # últimos 7 días, sin clave
    python run_firms.py --bbox 144.35,-19.75,144.85,-19.25 --recent 7d

    # cualquier fecha, con la MAP_KEY gratuita de FIRMS en FIRMS_MAP_KEY
    python run_firms.py --bbox 144.35,-19.75,144.85,-19.25 --start 2026-11-08 --days 7

    # sin red, desde CSV ya descargados (el nombre del archivo es la fuente)
    python run_firms.py --bbox 144.35,-19.75,144.85,-19.25 --csv-dir data/queensland_2026-11
"""
import argparse
import json
import os
from datetime import date

import numpy as np

from frp_enkf import firms, plots
from frp_enkf.calibrate import calibrate_from_detections
from frp_enkf.assim import FilterConfig, FireFilter, run_passes
from frp_enkf.model import FireParams, smooth_field


def load_detections(args, bbox) -> list[firms.Detections]:
    dets = []
    for source in args.sources:
        if args.csv_dir:
            path = os.path.join(args.csv_dir, f"{source}.csv")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        elif args.recent:
            text = firms.fetch_recent(source, args.recent, args.cache)
        else:
            text = firms.fetch_area(source, bbox, date.fromisoformat(args.start), args.days,
                                    args.map_key, args.cache)
        d = firms.parse_csv(text, source, bbox)
        print(f"{source:<18} {len(d):>6} detecciones")
        dets.append(d)
    return dets


def forecast_skill(log: list[dict]) -> dict:
    """Error del pronóstico del filtro al llegar cada pasada, contra persistencia.

    Para cada pasada compara la suma de FRP observada con lo que el filtro esperaba
    ver en esos mismos píxeles, y con la suma de la pasada anterior del mismo tipo
    de sensor (persistencia). Se excluye la primera pasada de cada tipo.
    """
    out = {}
    for fam in ("MODIS", "VIIRS"):
        rows = [r for r in log if r["family"] == fam and r["obs_sum"] > 0]
        if len(rows) < 3:
            continue
        obs = np.array([r["obs_sum"] for r in rows])
        pred = np.array([max(r["pred_sum"], 1e-3) for r in rows])
        out[fam] = {
            "pasadas": len(rows) - 1,
            "filtro_factor_mediano": float(np.exp(np.median(np.abs(np.log(pred[1:] / obs[1:]))))),
            "persistencia_factor_mediano": float(np.exp(np.median(np.abs(np.log(obs[:-1] / obs[1:]))))),
        }
    return out


def holdout_skill(rows: list[dict]) -> dict:
    """Resume la validación con el satélite reservado, para el filtro y la persistencia.

    factor_mediano: error mediano del FRP total de la pasada, como factor.
    csi: aciertos / (aciertos + omisiones + falsas alarmas) sobre píxeles con fuego.
    """
    out = {"pasadas": len(rows)}
    floor = 1.0   # MW, para que una pasada sin fuego predicho no dé un factor infinito
    for name in ("filtro", "persistencia"):
        ratio = [np.log((r[f"{name}_total"] + floor) / (r["obs_total"] + floor)) for r in rows]
        h = sum(r[f"{name}_hits"] for r in rows)
        m = sum(r[f"{name}_misses"] for r in rows)
        f = sum(r[f"{name}_false"] for r in rows)
        out[name] = {"factor_mediano": float(np.exp(np.median(np.abs(ratio)))),
                     "sesgo_mediano": float(np.exp(np.median(ratio))),
                     "csi": h / max(h + m + f, 1), "aciertos": h, "omisiones": m,
                     "falsas_alarmas": f}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bbox", required=True, help="oeste,sur,este,norte en grados")
    ap.add_argument("--start", help="primer día UTC, AAAA-MM-DD (API con MAP_KEY)")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--recent", choices=["24h", "48h", "7d"], help="archivos globales sin clave")
    ap.add_argument("--csv-dir", help="carpeta con <FUENTE>.csv ya descargados")
    ap.add_argument("--sources", nargs="+", default=list(firms.DEFAULT_SOURCES))
    ap.add_argument("--map-key", help="MAP_KEY de FIRMS (o variable FIRMS_MAP_KEY)")
    ap.add_argument("--cache", default=".firms_cache")
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--no-non-detections", action="store_true",
                    help="no asumir que los píxeles sin detección estaban despejados")
    ap.add_argument("--holdout", help="plataforma que no se asimila y se usa para validar, "
                    "por ejemplo NOAA-21")
    ap.add_argument("--no-calibration", action="store_true",
                    help="no calibrar x_peak y d_min con las detecciones")
    ap.add_argument("--nondet-level", type=float, default=1.0,
                    help="una no-detección significa FRP <= nivel x umbral de detección")
    ap.add_argument("--birth-fuel-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="salida_firms")
    args = ap.parse_args()
    if not (args.start or args.recent or args.csv_dir):
        ap.error("indica --start (con MAP_KEY), --recent o --csv-dir")

    bbox = tuple(float(v) for v in args.bbox.split(","))
    try:
        dets = [d for d in load_detections(args, bbox) if len(d)]
    except firms.FirmsError as exc:
        raise SystemExit(f"Error de FIRMS: {exc}")
    if not dets:
        raise SystemExit("No hay detecciones en ese rectángulo y período.")

    first = min(d.time.min() for d in dets)
    dom = firms.Domain(*bbox, t0=first.astype("datetime64[D]"))
    passes = firms.passes_from_detections(dets, dom, not args.no_non_detections)
    print(f"Grilla de {dom.shape[1]} x {dom.shape[0]} celdas de 375 m, {len(passes)} pasadas "
          f"({sum(o.sensor.family == 'MODIS' for o in passes)} MODIS, "
          f"{sum(o.sensor.family == 'VIIRS' for o in passes)} VIIRS)")

    dt_h = 1.0 / 6.0
    t_start = np.floor(passes[0].time_h) - 1.0
    times = t_start + dt_h * np.arange(int(np.ceil((passes[-1].time_h + 3.0 - t_start) / dt_h)) + 1)
    by_step: dict[int, list] = {}
    held: dict[int, list] = {}
    for o in passes:
        target = held if o.platform == args.holdout else by_step
        target.setdefault(max(1, int(round((o.time_h - t_start) / dt_h))), []).append(o)
    if args.holdout and not held:
        raise SystemExit(f"No hay pasadas de {args.holdout} para reservar.")

    model, calib = FireParams(), {}
    if not args.no_calibration:
        model, calib = calibrate_from_detections(dets)
        if "x_peak_mw" in calib:
            print(f"Calibración: x_peak = {model.x_peak:.1f} MW, d_min = {model.d_min:.2f}")

    rng = np.random.default_rng(args.seed)
    # sin mapa de combustible real: uno uniforme con incertidumbre amplia
    fuel = np.clip(0.75 + 0.2 * smooth_field(dom.shape, 5.0, rng, args.members), 0.2, 1.0)
    has_viirs = any(o.sensor.family == "VIIRS" for ps in by_step.values() for o in ps)
    # sin máscara de nubes, una no-detección vale menos que en el experimento sintético;
    # y como el modelo no está calibrado para este incendio, sus tasas se ajustan en línea
    cfg = FilterConfig(sigma_nondet=0.6, estimate_bias=has_viirs, estimate_params=True,
                       nondet_level=args.nondet_level,
                       birth_fuel_weight=args.birth_fuel_weight)
    flt = FireFilter(fuel, model.perturbed(args.members, rng, rel=0.3, wind_sd=0.3), rng, cfg)
    res = run_passes(flt, by_step, times, "FIRMS", store_every=6, holdout=held)

    os.makedirs(args.out, exist_ok=True)
    tot = res.total_members
    lo, hi = np.percentile(tot, [5, 95], axis=1)
    with open(os.path.join(args.out, "frp_armonizado.csv"), "w", encoding="utf-8") as fh:
        fh.write("utc,frp_total_mw,p05_mw,p95_mw\n")
        for k in range(0, len(times), 6):
            fh.write(f"{dom.utc(times[k])},{tot[k].mean():.1f},{lo[k]:.1f},{hi[k]:.1f}\n")
    np.savez_compressed(os.path.join(args.out, "frp_armonizado.npz"), frp_mw=res.mean.astype("f4"),
                        dispersion_mw=res.spread.astype("f4"), horas_locales=res.times,
                        utc=np.array([str(dom.utc(t)) for t in res.times]), bbox=np.array(bbox))
    skill = forecast_skill(res.log)
    summary = {"bbox": bbox, "pasadas": len(passes), "calibracion": calib,
               "habilidad_pronostico": skill,
               "frp_total_maximo_mw": float(tot.mean(axis=1).max()),
               "suma_maxima_de_una_pasada_mw": float(max(np.nansum(o.frp) for o in passes))}
    print(f"FRP total máximo del filtro: {summary['frp_total_maximo_mw']:.0f} MW; "
          f"mayor suma de una pasada: {summary['suma_maxima_de_una_pasada_mw']:.0f} MW")
    if res.verification:
        summary["validacion_" + args.holdout] = holdout_skill(res.verification)
        for name, s in summary["validacion_" + args.holdout].items():
            if isinstance(s, dict):
                print(f"Validación con {args.holdout}, {name}: error del total ×{s['factor_mediano']:.2f}, "
                      f"CSI {s['csi']:.2f} (aciertos {s['aciertos']}, omisiones {s['omisiones']}, "
                      f"falsas alarmas {s['falsas_alarmas']})")
    if res.bias_trace:
        _, b, sd = res.bias_trace[-1]
        summary["sesgo_modis_sobre_viirs"] = {"b": b, "desvio_log": sd}
        print(f"Sesgo MODIS/VIIRS estimado: {b:.2f} (±{sd:.2f} en log)")
    for fam, s in skill.items():
        print(f"Pronóstico {fam}: error mediano ×{s['filtro_factor_mediano']:.2f} "
              f"(persistencia ×{s['persistencia_factor_mediano']:.2f}, {s['pasadas']} pasadas)")
    with open(os.path.join(args.out, "resumen.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    plots.fig_real(dom, passes, res, os.path.join(args.out, "frp_armonizado.png"))
    print(f"Resultados en {args.out}/")


if __name__ == "__main__":
    main()
