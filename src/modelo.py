"""
Módulo: Modelo del Dominio
===========================
Contenido:
  - Tablas del enunciado (DEATH_PROB, PREGNANCY_PROB, PARTNER_PROB, GRIEF_LAMBDA)
  - Clase Person
  - generate_population()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from src.generadores import RandomVariables


# ─────────────────────────────────────────────────────────────────────────────
# TABLAS DEL ENUNCIADO
# ─────────────────────────────────────────────────────────────────────────────

# (edad_min, edad_max, probabilidad)
DEATH_PROB = {
    "M": [(0, 12, 0.25), (12, 45, 0.10), (45, 76, 0.30), (76, 125, 0.70)],
    "F": [(0, 12, 0.25), (12, 45, 0.15), (45, 76, 0.35), (76, 125, 0.65)],
}

PREGNANCY_PROB = [
    (12, 15, 0.20), (15, 21, 0.45), (21, 35, 0.80),
    (35, 45, 0.40), (45, 60, 0.20), (60, 125, 0.05),
]

PARTNER_PROB = [
    (12, 15, 0.60), (15, 21, 0.65), (21, 35, 0.80),
    (35, 45, 0.60), (45, 60, 0.50), (60, 125, 0.20),
]

# λ del duelo en unidades de 1/año  (e.g. 3 meses → λ = 12/3 = 4)
GRIEF_LAMBDA = [
    (12, 15,  4.00),   # 3 meses
    (15, 21,  2.00),   # 6 meses
    (21, 35,  2.00),   # 6 meses
    (35, 45,  1.00),   # 1 año
    (45, 60,  0.50),   # 2 años
    (60, 125, 0.25),   # 4 años
]

# Número máximo de hijos deseados — probabilidades brutas del enunciado
_CHILDREN_RAW = [(1, 0.60), (2, 0.75), (3, 0.35), (4, 0.20), (5, 0.10), (6, 0.05)]
_TOTAL        = sum(p for _, p in _CHILDREN_RAW)
CHILDREN_VALUES = [v for v, _ in _CHILDREN_RAW]
CHILDREN_PROBS  = [p / _TOTAL for _, p in _CHILDREN_RAW]   # normalizadas


def _lookup(age: float, table: list) -> Optional[float]:
    """Devuelve el valor del primer rango [lo, hi) que contiene `age`."""
    for lo, hi, val in table:
        if lo <= age < hi:
            return val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PERSONA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Person:
    """Entidad base de la simulación."""
    pid          : int
    sex          : str            # 'M' | 'F'
    age          : float          # años en t=0
    partner      : Optional[Person] = field(default=None, repr=False)
    children     : int   = 0
    max_children : int   = 2
    in_grief     : bool  = False
    alive        : bool  = True

    # ── Consultas de tabla ───────────────────────────────────────────────────

    def death_prob(self) -> Optional[float]:
        return _lookup(self.age, DEATH_PROB[self.sex])

    def pregnancy_prob(self) -> Optional[float]:
        if self.sex != "F":
            return None
        return _lookup(self.age, PREGNANCY_PROB)

    def wants_partner_prob(self) -> Optional[float]:
        return _lookup(self.age, PARTNER_PROB)

    def grief_lambda(self) -> Optional[float]:
        return _lookup(self.age, GRIEF_LAMBDA)

    # ── Estado ───────────────────────────────────────────────────────────────

    def is_single(self) -> bool:
        return self.partner is None and not self.in_grief

    def age_range_label(self) -> str:
        for lo, hi, _ in DEATH_PROB[self.sex]:
            if lo <= self.age < hi:
                return f"{lo}-{hi}"
        return "desconocido"

    def __hash__(self):
        return hash(self.pid)


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE POBLACIÓN INICIAL
# ─────────────────────────────────────────────────────────────────────────────

def generate_population(M: int, H: int, rng: RandomVariables) -> list[Person]:
    """
    Crea M mujeres y H hombres con:
      - edad ~ U(0, 100)
      - max_children sorteado con la tabla del enunciado
    """
    population: list[Person] = []
    pid = 0
    for sex, count in [("F", M), ("M", H)]:
        for _ in range(count):
            person = Person(
                pid          = pid,
                sex          = sex,
                age          = rng.uniform(0.0, 100.0),
                max_children = rng.discrete_choice(CHILDREN_VALUES, CHILDREN_PROBS),
            )
            population.append(person)
            pid += 1
    return population
