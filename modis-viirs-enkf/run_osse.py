#!/usr/bin/env python3
"""Corre el experimento completo y guarda métricas y figuras.

    python run_osse.py                 # escenario por defecto
    python run_osse.py --seed 3 --members 100 --out figures_seed3
"""
import argparse
import json
import os

from frp_enkf import plots
from frp_enkf.osse import Scenario, run_filter, run_naive, score, simulate_truth
from frp_enkf.sensors import MODIS

CONFIGS = [("MODIS + VIIRS", ("MODIS", "VIIRS")), ("Solo VIIRS", ("VIIRS",)),
           ("Solo MODIS", ("MODIS",))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--hours", type=float, default=72.0)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    sc = Scenario(seed=args.seed, hours=args.hours)
    truth = simulate_truth(sc)
    # las métricas arrancan 6 h después de la primera pasada (el filtro empieza a ciegas)
    t_start = truth.times[min(truth.passes)] + 6.0

    scores, runs = {}, {}
    for name, fam in CONFIGS:
        runs[name] = (run_filter(sc, truth, fam, name, n_members=args.members),
                      run_naive(sc, truth, fam, name))
        scores[f"Filtro · {name}"] = score(sc, truth, runs[name][0], t_start)
        scores[f"Base · {name}"] = score(sc, truth, runs[name][1], t_start)

    cols = ["rmse_375m", "rmse_1500m", "total_mae_pct", "total_bias_pct", "spread_over_rmse"]
    print(f"{'configuración':<24}" + "".join(f"{c:>18}" for c in cols))
    for name, s in scores.items():
        print(f"{name:<24}" + "".join(f"{s[c]:>18.2f}" if c in s else f"{'—':>18}" for c in cols))
    both = runs["MODIS + VIIRS"][0]
    b_hat, b_sd = both.bias_trace[-1][1:]
    print(f"\nSesgo MODIS/VIIRS estimado: {b_hat:.3f} (±{b_sd:.3f} en log); real: {MODIS.bias:.2f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump({"seed": args.seed, "members": args.members, "scores": scores,
                   "bias_estimate": b_hat, "bias_log_sd": b_sd, "bias_true": MODIS.bias},
                  fh, indent=2, ensure_ascii=False)
    if not args.no_figures:
        plots.fig_problem(truth, os.path.join(args.out, "1_problema.png"))
        plots.fig_total(truth, both, runs["MODIS + VIIRS"][1], os.path.join(args.out, "2_frp_total.png"))
        plots.fig_maps(sc, truth, both, os.path.join(args.out, "3_mapas.png"))
        plots.fig_scores(scores, both, MODIS.bias, os.path.join(args.out, "4_metricas.png"))
        print(f"Figuras en {args.out}/")


if __name__ == "__main__":
    main()
