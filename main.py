"""
Poblado en Evolución — Punto de Entrada
========================================
Uso:
    python main.py                        # parámetros por defecto
    python main.py --seed 7 --m 300 --h 300 --years 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.generadores import LCG, RandomVariables
from src.modelo import generate_population
from src.motor_des import DESEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Simulación: Poblado en Evolución")
    parser.add_argument("--seed",  type=int,   default=42,    help="Semilla LCG")
    parser.add_argument("--m",     type=int,   default=200,   help="Número de mujeres iniciales")
    parser.add_argument("--h",     type=int,   default=200,   help="Número de hombres iniciales")
    parser.add_argument("--years", type=float, default=100.0, help="Horizonte de simulación (años)")
    return parser.parse_args()


def main():
    args = parse_args()

    rng        = RandomVariables(LCG(seed=args.seed))
    population = generate_population(args.m, args.h, rng)
    engine     = DESEngine(population, rng, sim_horizon=args.years)

    engine.run()
    engine.report()


if __name__ == "__main__":
    main()
