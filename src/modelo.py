"""
Módulo: Modelo del Dominio

Contenido:
  - Tablas del enunciado (DEATH_PROB, PREGNANCY_PROB, PARTNER_PROB, GRIEF_LAMBDA)
  - Clase Person
  - generate_population()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from src.generadores import RandomVariables


# ======================
# TABLAS DEL ENUNCIADO =
# ======================

# Tabla de probabilidad de muerte por sexo y rango de edad.
# Cada tupla tiene la forma: (edad_min, edad_max, probabilidad)
DEATH_PROB = {
    "M": [(0, 12, 0.25), (12, 45, 0.10), (45, 76, 0.30), (76, 125, 0.70)],
    "F": [(0, 12, 0.25), (12, 45, 0.15), (45, 76, 0.35), (76, 125, 0.65)],
}

# Tabla de probabilidad de embarazo según rango de edad.
# Solo aplica a personas de sexo femenino.
PREGNANCY_PROB = [
    (12, 15, 0.20), (15, 21, 0.45), (21, 35, 0.80),
    (35, 45, 0.40), (45, 60, 0.20), (60, 125, 0.05),
]

# Tabla de probabilidad de querer pareja según rango de edad.
PARTNER_PROB = [
    (12, 15, 0.60), (15, 21, 0.65), (21, 35, 0.80),
    (35, 45, 0.60), (45, 60, 0.50), (60, 125, 0.20),
]

# λ del duelo en unidades de 1/año.
# Ejemplo: 3 meses equivalen a λ = 4.
GRIEF_LAMBDA = [
    (12, 15,  4.00),   # 3 meses
    (15, 21,  2.00),   # 6 meses
    (21, 35,  2.00),   # 6 meses
    (35, 45,  1.00),   # 1 año
    (45, 60,  0.50),   # 2 años
    (60, 125, 0.25),   # 4 años
]

# ***** Distribución de hijos deseados *****
# Distribución discreta del enunciado.
# Cada valor representa el número máximo de hijos que una persona desea tener.
CHILDREN_VALUES = [1,    2,    3,    4,    5,    6   ]
CHILDREN_PROBS  = [0.10, 0.30, 0.40, 0.10, 0.05, 0.05]   # suma = 1.0


def sample_max_children(rng: RandomVariables) -> int:
    """
    Muestrea el número máximo de hijos deseados desde la distribución
    discreta del enunciado mediante transformada inversa.
    """
    return rng.discrete_choice(CHILDREN_VALUES, CHILDREN_PROBS)


# Probabilidad de establecer pareja según diferencia de edad.
AGE_DIFF_PROB = [
    (0,   5,   0.45),
    (5,   10,  0.40),
    (10,  15,  0.35),
    (15,  20,  0.25),
    (20,  200, 0.15),
]

# Distribución del número de bebés en un parto múltiple.
# Primero se definen probabilidades crudas y luego se normalizan.
_MB_RAW               = [0.70, 0.18, 0.08, 0.04, 0.02]
_MB_TOTAL             = sum(_MB_RAW)
MULTIPLE_BIRTH_VALUES = [1, 2, 3, 4, 5]
MULTIPLE_BIRTH_PROBS  = [p / _MB_TOTAL for p in _MB_RAW]

# Parámetros globales de la simulación
BREAKUP_PROB        = 0.20  # probabilidad de ruptura por chequeo
PREGNANCY_DURATION  = 0.75  # duración del embarazo (~9 meses en años)
PARTNER_ATTEMPT_LAM = 3.0   # Exp(3): intento de pareja cada ~4 meses
PREGNANCY_CHECK_LAM = 4.0   # Exp(4): chequeo de embarazo cada ~3 meses
BREAKUP_CHECK_LAM   = 0.5   # Exp(0.5): chequeo de ruptura cada ~2 años


def _lookup(age: float, table: list) -> Optional[float]:
    """Devuelve el valor del primer rango [lo, hi) que contiene `age`."""
    for lo, hi, val in table:
        if lo <= age < hi:
            return val
    return None


# ==========
# PERSONA  =
# ==========

@dataclass
class Person:
    """Entidad base de la simulación."""
    # Identificador único de la persona
    pid          : int
    # Sexo biológico usado por las reglas del modelo
    sex          : str            # 'M' | 'F'
    # Edad actual de la persona
    age          : float          # edad actual (se actualiza en cada evento)
    # Instante de simulación en que nació
    # Para población inicial: birth_time = -age_inicial
    birth_time   : float = 0.0
    # Pareja actual, si existe
    partner      : Optional[Person] = field(default=None, repr=False)
    # Número de hijos ya nacidos
    children     : int   = 0
    # Número máximo de hijos deseados
    max_children : int   = 2
    # Estado de duelo tras una ruptura o pérdida
    in_grief     : bool  = False
    # Estado de embarazo
    pregnant     : bool  = False   # True durante los ~9 meses de gestación
    # Estado de vida
    alive        : bool  = True

    def age_at(self, t: float) -> float:
        """Edad exacta en el instante t de simulación."""
        return t - self.birth_time

    # ***** Consultas de tabla *****

    def death_prob(self) -> Optional[float]:
        # Probabilidad de morir según sexo y edad actual
        return _lookup(self.age, DEATH_PROB[self.sex])

    def pregnancy_prob(self) -> Optional[float]:
        # Solo las mujeres pueden tener probabilidad de embarazo
        if self.sex != "F":
            return None
        return _lookup(self.age, PREGNANCY_PROB)

    def wants_partner_prob(self) -> Optional[float]:
        # Probabilidad de buscar pareja según la edad
        return _lookup(self.age, PARTNER_PROB)

    def grief_lambda(self) -> Optional[float]:
        # Tasa del proceso de duelo según edad
        return _lookup(self.age, GRIEF_LAMBDA)

    # ***** Estado *****

    def is_single(self) -> bool:
        # Está sola/o si no tiene pareja y tampoco está en duelo
        return self.partner is None and not self.in_grief

    def age_range_label(self) -> str:
        # Devuelve la etiqueta textual del rango de edad actual
        for lo, hi, _ in DEATH_PROB[self.sex]:
            if lo <= self.age < hi:
                return f"{lo}-{hi}"
        return "desconocido"

    def __hash__(self):
        # Permite usar Person en estructuras hashables como sets o dicts
        return hash(self.pid)


# ==================================
# GENERACIÓN DE POBLACIÓN INICIAL  =
# ==================================

def generate_population(M: int, H: int, rng: RandomVariables) -> list[Person]:
    """
    Crea M mujeres y H hombres con:
      - edad ~ U(0, 100)
      - max_children sorteado con una distribución discreta
    """
    population: list[Person] = []
    pid = 0

    # Se generan primero las mujeres y luego los hombres
    for sex, count in [("F", M), ("M", H)]:
        for _ in range(count):
            # Edad inicial uniforme entre 0 y 100
            age = rng.uniform(0.0, 100.0)

            # Se crea la persona con su estado inicial
            person = Person(
                pid          = pid,
                sex          = sex,
                age          = age,
                birth_time   = -age,    # nació `age` años antes del inicio
                max_children = sample_max_children(rng),   # número máximo de hijos deseados
            )

            # Se agrega a la población total
            population.append(person)
            pid += 1

    return population