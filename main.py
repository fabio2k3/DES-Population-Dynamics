"""
Poblado en Evolución — Punto de Entrada

Uso:
    python main.py                              # 1 corrida, parámetros por defecto
    python main.py --runs 10                    # 10 corridas, análisis estadístico
    python main.py --seed 7 --m 300 --h 300
    python main.py --runs 10 --no-plots         # sin generar imágenes
"""

import argparse
import sys
import os

# Permite ejecutar el proyecto desde la raíz sin problemas de importación
sys.path.insert(0, os.path.dirname(__file__))

# Importa los componentes principales del proyecto
from src.generadores import LCG, RandomVariables
from src.modelo import generate_population
from src.motor_des import DESEngine
from src.estadisticas import StatsCollector, MultiRunAnalysis
from src.visualizacion import plot_all


def parse_args():
    # Configura el analizador de argumentos de línea de comandos
    p = argparse.ArgumentParser(description="Simulación: Poblado en Evolución")

    # Semilla base para el generador pseudoaleatorio
    p.add_argument("--seed", type=int, default=42, help="Semilla LCG base")
    # Cantidad inicial de mujeres
    p.add_argument("--m", type=int, default=200, help="Mujeres iniciales")
    # Cantidad inicial de hombres
    p.add_argument("--h", type=int, default=200, help="Hombres iniciales")
    # Horizonte temporal de la simulación
    p.add_argument("--years", type=float, default=100.0, help="Horizonte (años)")
    # Número de corridas independientes
    p.add_argument("--runs", type=int, default=1, help="Número de corridas")
    # Opción para desactivar la generación de gráficas
    p.add_argument("--no-plots", action="store_true", help="No generar gráficas")

    return p.parse_args()


def run_once(seed: int, M: int, H: int, years: float,
             verbose: bool = True) -> tuple[DESEngine, object]:
    # Se crea un generador aleatorio basado en LCG con la semilla indicada
    rng = RandomVariables(LCG(seed=seed))

    # Se genera la población inicial con M mujeres y H hombres
    population = generate_population(M, H, rng)

    # Se instancia el motor DES con la población y el horizonte dado
    engine = DESEngine(population, rng, sim_horizon=years)

    # Ejecuta la simulación completa
    engine.run()

    # Si verbose está activo, imprime el reporte final del motor
    if verbose:
        engine.report()

    # Extrae estadísticas resumidas de la corrida
    collector = StatsCollector(engine, seed=seed)
    summary = collector.collect()

    # Devuelve el motor y el resumen de la simulación
    return engine, summary


def main():
    # Lee y procesa los argumentos del usuario
    args = parse_args()

    # Caso de una sola corrida
    if args.runs == 1:
        engine, summary = run_once(args.seed, args.m, args.h, args.years)

        # Genera las gráficas solo si el usuario no pidió desactivarlas
        if not args.no_plots:
            plot_all(summary, multi=None, output_dir="resultados/")

    else:
        # Caso de múltiples corridas para análisis estadístico
        print(f"\n  Ejecutando {args.runs} corridas independientes...\n")

        summaries = []
        for i in range(args.runs):
            # Cada corrida usa una semilla distinta derivada de la base
            seed = args.seed + i
            print(f"  Corrida {i+1}/{args.runs}  (seed={seed})")

            # Solo la primera corrida imprime el reporte completo
            _, summary = run_once(seed, args.m, args.h, args.years,
                                  verbose=(i == 0))
            summaries.append(summary)

        # Construye el análisis agregado de todas las corridas
        multi = MultiRunAnalysis(n_runs=args.runs, summaries=summaries)
        multi.print_summary()

        # Genera las gráficas usando la primera corrida y el análisis global
        if not args.no_plots:
            plot_all(summaries[0], multi=multi, output_dir="resultados/")


if __name__ == "__main__":
    # Punto de entrada del programa
    main()