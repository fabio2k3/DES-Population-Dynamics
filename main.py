"""
Poblado en Evolución — Punto de Entrada
========================================
Uso:
    python main.py                              # 1 corrida, parámetros por defecto
    python main.py --runs 10                    # 10 corridas, análisis estadístico
    python main.py --seed 7 --m 300 --h 300
    python main.py --runs 10 --no-plots         # sin generar imágenes
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.generadores import LCG, RandomVariables
from src.modelo import generate_population
from src.motor_des import DESEngine
from src.estadisticas import StatsCollector, MultiRunAnalysis
from src.visualizacion import plot_all


def parse_args():
    p = argparse.ArgumentParser(description="Simulación: Poblado en Evolución")
    p.add_argument("--seed",     type=int,   default=42,    help="Semilla LCG base")
    p.add_argument("--m",        type=int,   default=200,   help="Mujeres iniciales")
    p.add_argument("--h",        type=int,   default=200,   help="Hombres iniciales")
    p.add_argument("--years",    type=float, default=100.0, help="Horizonte (años)")
    p.add_argument("--runs",     type=int,   default=1,     help="Número de corridas")
    p.add_argument("--no-plots", action="store_true",       help="No generar gráficas")
    return p.parse_args()


def run_once(seed: int, M: int, H: int, years: float,
             verbose: bool = True) -> tuple[DESEngine, object]:
    rng        = RandomVariables(LCG(seed=seed))
    population = generate_population(M, H, rng)
    engine     = DESEngine(population, rng, sim_horizon=years)
    engine.run()
    if verbose:
        engine.report()
    collector = StatsCollector(engine, seed=seed)
    summary   = collector.collect()
    return engine, summary


def main():
    args = parse_args()

    if args.runs == 1:
        engine, summary = run_once(args.seed, args.m, args.h, args.years)

        if not args.no_plots:
            plot_all(summary, multi=None, output_dir="resultados/")

    else:
        print(f"\n  Ejecutando {args.runs} corridas independientes...\n")
        summaries = []
        for i in range(args.runs):
            seed = args.seed + i
            print(f"  Corrida {i+1}/{args.runs}  (seed={seed})")
            _, summary = run_once(seed, args.m, args.h, args.years,
                                  verbose=(i == 0))
            summaries.append(summary)

        multi = MultiRunAnalysis(n_runs=args.runs, summaries=summaries)
        multi.print_summary()

        if not args.no_plots:
            plot_all(summaries[0], multi=multi, output_dir="resultados/")


if __name__ == "__main__":
    main()