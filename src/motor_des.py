"""
Módulo: Motor de Simulación de Eventos Discretos
=================================================
Contenido:
  - EventType  : enumeración de todos los eventos del proyecto
  - Event      : elemento de la FEL (min-heap)
  - DESEngine  : motor principal (FEL + manejadores + estadísticas)
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto

from src.generadores import RandomVariables
from src.modelo import Person, DEATH_PROB, generate_population



# TIPOS DE EVENTO
class EventType(Enum):
    DEATH           = auto()
    AGE_TRANSITION  = auto()
    # Reservados para módulos futuros
    GRIEF_END       = auto()
    PARTNER_ATTEMPT = auto()
    BREAKUP         = auto()
    PREGNANCY       = auto()
    BIRTH           = auto()


# EVENTO
@dataclass(order=True)
class Event:
    """
    Elemento de la FEL ordenado por tiempo.
    `sort_index` rompe empates de forma FIFO.
    """
    time        : float
    sort_index  : int         = field(compare=True)
    event_type  : EventType   = field(compare=False)
    person      : Person      = field(compare=False)


# MOTOR DES
class DESEngine:
    """
    Motor de Simulación de Eventos Discretos.

    El reloj avanza de evento en evento (no en pasos fijos).
    La FEL es un min-heap ordenado por tiempo de evento.
    """

    def __init__(self, population: list[Person], rng: RandomVariables,
                 sim_horizon: float = 100.0):
        self.population  = population
        self.rng         = rng
        self.horizon     = sim_horizon
        self.clock       = 0.0
        self._fel        : list[Event] = []
        self._counter    = 0              # desempate FIFO

        # Estadísticas
        self.total_deaths        = 0
        self.deaths_by_age_range = defaultdict(int)
        self.deaths_by_sex       = defaultdict(int)
        self.population_snapshot = []     # (año, n_vivos)
        self._next_snapshot      = 10.0

    # FEL 
    def schedule(self, time: float, event_type: EventType, person: Person) -> None:
        """Inserta un evento en la FEL. Descarta eventos fuera del horizonte."""
        if time > self.horizon:
            return
        heapq.heappush(self._fel, Event(
            time       = time,
            sort_index = self._counter,
            event_type = event_type,
            person     = person,
        ))
        self._counter += 1

    # Lógica de supervivencia
    def _schedule_fate(self, person: Person, entry_time: float) -> None:
        """
        Decide el destino de una persona al entrar a un rango de edad:
          - Con prob p    → DEATH en tiempo uniforme dentro del rango.
          - Con prob 1-p  → AGE_TRANSITION al cumplir el límite superior.
          - Sin rango     → DEATH inmediata (superó los 125 años).
        """
        current = None
        for lo, hi, prob in DEATH_PROB[person.sex]:
            if lo <= person.age < hi:
                current = (lo, hi, prob)
                break

        if current is None:
            self.schedule(entry_time, EventType.DEATH, person)
            return

        lo, hi, prob = current
        years_left = hi - person.age

        if self.rng.bernoulli(prob):
            t_death = self.rng.uniform(0.0, years_left)
            self.schedule(entry_time + t_death, EventType.DEATH, person)
        else:
            self.schedule(entry_time + years_left, EventType.AGE_TRANSITION, person)

    # Manejadores 
    def handle_death(self, evt: Event) -> None:
        person = evt.person
        if not person.alive:
            return

        person.alive = False
        person.age   = person.age + (evt.time - self.clock)

        if person.partner is not None:
            partner         = person.partner
            person.partner  = None
            partner.partner = None
            partner.in_grief = True
            lam = partner.grief_lambda()
            if lam:
                self.schedule(
                    evt.time + self.rng.exponential(lam),
                    EventType.GRIEF_END, partner
                )

        self.total_deaths += 1
        self.deaths_by_sex[person.sex] += 1
        self.deaths_by_age_range[person.age_range_label()] += 1

    def handle_age_transition(self, evt: Event) -> None:
        person = evt.person
        if not person.alive:
            return
        person.age = evt.time
        self._schedule_fate(person, evt.time)

    # Snapshots 
    def _snapshot(self) -> None:
        alive = sum(1 for p in self.population if p.alive)
        self.population_snapshot.append((round(self.clock, 1), alive))

    # Loop principal
    def run(self) -> None:
        """Ejecuta la simulación hasta `self.horizon`."""
        print(f"\n{'═'*55}")
        print(f"  Simulación iniciada  |  horizonte = {self.horizon:.0f} años")
        print(f"  Población inicial    |  {len(self.population)} personas")
        print(f"{'═'*55}\n")

        for person in self.population:
            self._schedule_fate(person, entry_time=0.0)

        self._snapshot()

        processed = 0
        while self._fel:
            evt = heapq.heappop(self._fel)
            if evt.time > self.horizon:
                break

            self.clock = evt.time

            if self.clock >= self._next_snapshot:
                self._snapshot()
                self._next_snapshot += 10.0

            if evt.event_type == EventType.DEATH:
                self.handle_death(evt)
            elif evt.event_type == EventType.AGE_TRANSITION:
                self.handle_age_transition(evt)
            # GRIEF_END, PREGNANCY, etc. → módulos futuros

            processed += 1

        self.clock = self.horizon
        self._snapshot()
        print(f"  Eventos procesados : {processed}")
        print(f"  Reloj final        : {self.clock:.1f} años\n")

    #  Reporte
    def report(self) -> None:
        alive   = [p for p in self.population if p.alive]
        alive_f = sum(1 for p in alive if p.sex == "F")
        alive_m = sum(1 for p in alive if p.sex == "M")
        sep = "─" * 55

        print(f"\n{sep}")
        print("  REPORTE FINAL — Poblado en Evolución")
        print(sep)
        print(f"  Horizonte          : {self.horizon:.0f} años")
        print(f"  Población inicial  : {len(self.population)}")
        print(f"  Vivos al final     : {len(alive)}  (F={alive_f}, M={alive_m})")
        print(f"  Fallecidos totales : {self.total_deaths}")

        print(f"\n  Muertes por sexo:")
        for sex, n in sorted(self.deaths_by_sex.items()):
            print(f"    {'Mujeres' if sex=='F' else 'Hombres':10s}: {n}")

        print(f"\n  Muertes por rango de edad:")
        for rango in ["0-12", "12-45", "45-76", "76-125"]:
            n   = self.deaths_by_age_range.get(rango, 0)
            bar = "█" * min(n, 40)
            print(f"    {rango:>8s} → {n:4d}  {bar}")

        print(f"\n  Evolución de la población:")
        print(f"  {'Año':>5}  {'Vivos':>6}")
        print(f"  {'─'*14}")
        for year, n in self.population_snapshot:
            bar = "█" * (n * 30 // max(1, len(self.population)))
            print(f"  {year:>5.0f}  {n:>6}  {bar}")
        print(sep)
