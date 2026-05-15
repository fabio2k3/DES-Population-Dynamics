"""
Módulo: Motor de Simulación de Eventos Discretos

Eventos: DEATH, AGE_TRANSITION, GRIEF_END,
         PARTNER_ATTEMPT, BREAKUP, PREGNANCY, BIRTH
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto

# Importa los generadores aleatorios del proyecto
from src.generadores import RandomVariables

# Importa el modelo del dominio y constantes necesarias para la simulación
from src.modelo import (
    Person, DEATH_PROB, AGE_DIFF_PROB,
    MULTIPLE_BIRTH_VALUES, MULTIPLE_BIRTH_PROBS,
    BREAKUP_PROB, BREAKUP_CHECK_LAM,        # Constante de chequeo de ruptura
    PREGNANCY_DURATION,
    PARTNER_ATTEMPT_LAM, PREGNANCY_CHECK_LAM,
    sample_max_children,                    # Muestreo discreto del máximo de hijos
)


# ===================
# TIPOS DE EVENTO   =
# ===================

class EventType(Enum):
    # Muerte de una persona
    DEATH           = auto()
    # Cambio de rango de edad
    AGE_TRANSITION  = auto()
    # Fin del duelo
    GRIEF_END       = auto()
    # Intento de buscar pareja
    PARTNER_ATTEMPT = auto()
    # Posible ruptura de pareja
    BREAKUP         = auto()
    # Revisión de embarazo
    PREGNANCY       = auto()
    # Nacimiento de uno o más bebés
    BIRTH           = auto()


# =========
# EVENTO  =
# =========

@dataclass(order=True)
class Event:
    """Elemento de la FEL ordenado por tiempo. sort_index rompe empates FIFO."""
    # Instante en que ocurre el evento
    time        : float
    # Índice auxiliar para conservar orden FIFO entre eventos con mismo tiempo
    sort_index  : int         = field(compare=True)
    # Tipo de evento que se procesará
    event_type  : EventType   = field(compare=False)
    # Persona asociada al evento
    person      : Person      = field(compare=False)


# ============
# MOTOR DES  =
# ============

class DESEngine:
    # Cada cuántos eventos limpiar los sets de solteros de entradas obsoletas
    _CLEANUP_INTERVAL = 2_000

    def __init__(self, population: list[Person], rng: RandomVariables,
                 sim_horizon: float = 100.0):
        # Población total de la simulación
        self.population  = population
        # Generador aleatorio compartido
        self.rng         = rng
        # Horizonte temporal de la simulación
        self.horizon     = sim_horizon
        # Reloj de simulación
        self.clock       = 0.0
        # Future Event List (FEL) implementada con heap
        self._fel        : list[Event] = []
        # Contador interno para ordenar eventos con el mismo tiempo
        self._counter    = 0
        # Siguiente identificador disponible para nuevos nacidos
        self._next_pid   = max(p.pid for p in population) + 1
        # Contador de eventos desde el último cleanup
        self._events_since_cleanup = 0

        # Conjuntos de personas solteras elegibles para buscar pareja
        self._singles_f  : set[Person] = set()
        self._singles_m  : set[Person] = set()

        # Estadísticas globales acumuladas
        self.total_deaths         = 0
        self.total_births         = 0
        self.total_couples_formed = 0
        self.total_breakups       = 0
        self.deaths_by_age_range  = defaultdict(int)
        self.deaths_by_sex        = defaultdict(int)
        # Lista de snapshots de población a lo largo del tiempo
        self.population_snapshot  = []
        # Próximo instante en el que se toma snapshot
        self._next_snapshot       = 10.0

    # ***** FEL *****

    def schedule(self, time: float, event_type: EventType, person: Person) -> None:
        # No agenda eventos que ocurren fuera del horizonte
        if time > self.horizon:
            return
        # Inserta el evento en la cola prioritaria
        heapq.heappush(self._fel, Event(
            time=time, sort_index=self._counter,
            event_type=event_type, person=person,
        ))
        self._counter += 1

    # ***** Helpers *****

    def _new_pid(self) -> int:
        # Genera un nuevo identificador único para una persona nacida en sim
        pid = self._next_pid
        self._next_pid += 1
        return pid

    def _is_eligible_single(self, person: Person) -> bool:
        """Una persona es elegible para buscar pareja si: viva, soltera,
        sin duelo y tiene al menos 12 años."""
        return (person.alive
                and person.is_single()
                and person.age_at(self.clock) >= 12)

    def _register_single(self, person: Person) -> None:
        """Añade al pool de solteros si cumple todos los criterios."""
        if self._is_eligible_single(person):
            (self._singles_f if person.sex == "F" else self._singles_m).add(person)

    def _unregister_single(self, person: Person) -> None:
        # Elimina una persona de ambos pools por seguridad
        self._singles_f.discard(person)
        self._singles_m.discard(person)

    def _cleanup_singles(self) -> None:
        """
        Elimina de los sets personas que ya no son elegibles
        (muertas, en pareja, en duelo). Se ejecuta periódicamente.
        """
        self._singles_f = {p for p in self._singles_f if self._is_eligible_single(p)}
        self._singles_m = {p for p in self._singles_m if self._is_eligible_single(p)}

    def _age_diff_prob(self, age_a: float, age_b: float) -> float:
        # Calcula la probabilidad de compatibilidad según la diferencia de edad
        diff = abs(age_a - age_b)
        for lo, hi, prob in AGE_DIFF_PROB:
            if lo <= diff < hi:
                return prob
        return 0.15

    def _try_enter_partner_search(self, person: Person, t: float) -> None:
        """
        Punto único para registrar a una persona como soltera
        y agendar su primer PARTNER_ATTEMPT.
        Se usa desde GRIEF_END, AGE_TRANSITION y el inicio de la simulación.
        """
        if not self._is_eligible_single(person):
            return
        self._register_single(person)
        wp = person.wants_partner_prob()
        if wp:   # solo agenda si tiene alguna probabilidad de querer pareja
            self.schedule(
                t + self.rng.exponential(PARTNER_ATTEMPT_LAM),
                EventType.PARTNER_ATTEMPT, person
            )

    def _dissolve_couple(self, person: Person, t: float) -> None:
        """Disuelve la pareja, marca duelo y agenda GRIEF_END para ambos."""
        partner = person.partner
        if partner is None:
            return
        person.partner  = None
        partner.partner = None
        for p in (person, partner):
            if p.alive:
                p.in_grief = True
                self._unregister_single(p)
                lam = p.grief_lambda()
                if lam:
                    self.schedule(t + self.rng.exponential(lam),
                                  EventType.GRIEF_END, p)

    # ***** Supervivencia ******

    def _schedule_fate(self, person: Person, entry_time: float) -> None:
        """Decide DEATH o AGE_TRANSITION al entrar a un rango de edad."""
        # Actualiza la edad exacta al instante de entrada al rango
        person.age = person.age_at(entry_time)

        # Busca el intervalo actual de edad y su probabilidad de muerte
        current = None
        for lo, hi, prob in DEATH_PROB[person.sex]:
            if lo <= person.age < hi:
                current = (lo, hi, prob)
                break

        # Si no cae en ningún rango, agenda muerte inmediata como caso de borde
        if current is None:
            self.schedule(entry_time, EventType.DEATH, person)
            return

        lo, hi, prob = current
        years_left = hi - person.age

        # Con probabilidad 'prob' muere dentro del intervalo actual
        if self.rng.bernoulli(prob):
            self.schedule(entry_time + self.rng.uniform(0.0, years_left),
                          EventType.DEATH, person)
        else:
            # Si sobrevive al rango, agenda transición al siguiente umbral
            self.schedule(entry_time + years_left,
                          EventType.AGE_TRANSITION, person)

    # ***** Manejadores *****

    def handle_death(self, evt: Event) -> None:
        # Procesa la muerte de una persona
        person = evt.person
        if not person.alive:
            return
        person.alive = False
        person.age   = person.age_at(evt.time)
        self._unregister_single(person)
        if person.partner is not None:
            self._dissolve_couple(person, evt.time)
        self.total_deaths += 1
        self.deaths_by_sex[person.sex] += 1
        self.deaths_by_age_range[person.age_range_label()] += 1

    def handle_age_transition(self, evt: Event) -> None:
        """
        Al entrar al rango adulto, la persona puede pasar al pool de solteros
        si todavía está libre.
        """
        person = evt.person
        if not person.alive:
            return
        self._schedule_fate(person, evt.time)
        self._try_enter_partner_search(person, evt.time)

    def handle_grief_end(self, evt: Event) -> None:
        """
        Al terminar el duelo la persona vuelve al pool de solteros
        y agenda un PARTNER_ATTEMPT de forma garantizada.
        """
        person = evt.person
        if not person.alive:
            return
        person.in_grief = False
        self._try_enter_partner_search(person, evt.time)

    def handle_partner_attempt(self, evt: Event) -> None:
        """
        Intento de formar pareja:
          1. Verificar elegibilidad completa.
          2. Sortear deseo de pareja.
          3. Elegir candidato aleatorio del sexo opuesto.
          4. Sortear deseo de pareja del candidato.
          5. Sortear compatibilidad por diferencia de edad.
          6. Éxito → forman pareja, agenda BREAKUP + PREGNANCY.
          7. Fallo → re-agenda PARTNER_ATTEMPT.
        """
        person = evt.person
        if not self._is_eligible_single(person):
            self._unregister_single(person)
            return

        # Edad actual del solicitante en el instante del evento
        age = person.age_at(evt.time)
        # Probabilidad de querer pareja según rango de edad
        wp  = person.wants_partner_prob()
        if not wp or not self.rng.bernoulli(wp):
            self.schedule(evt.time + self.rng.exponential(PARTNER_ATTEMPT_LAM),
                          EventType.PARTNER_ATTEMPT, person)
            return

        # Pool del sexo opuesto
        pool     = self._singles_m if person.sex == "F" else self._singles_f
        eligible = [c for c in pool
                    if c.alive and c.is_single() and c is not person
                    and c.age_at(evt.time) >= 12]

        # Si no hay candidatos, se reintenta más adelante
        if not eligible:
            self.schedule(evt.time + self.rng.exponential(PARTNER_ATTEMPT_LAM),
                          EventType.PARTNER_ATTEMPT, person)
            return

        # Selección aleatoria uniforme entre los candidatos disponibles
        idx       = int(self.rng.uniform(0, len(eligible) - 1e-9))
        candidate = eligible[idx]
        age_c     = candidate.age_at(evt.time)

        # El candidato también debe querer pareja
        wp_c = candidate.wants_partner_prob()
        if not wp_c or not self.rng.bernoulli(wp_c):
            self.schedule(evt.time + self.rng.exponential(PARTNER_ATTEMPT_LAM),
                          EventType.PARTNER_ATTEMPT, person)
            return

        # Compatibilidad según diferencia de edad
        if not self.rng.bernoulli(self._age_diff_prob(age, age_c)):
            self.schedule(evt.time + self.rng.exponential(PARTNER_ATTEMPT_LAM),
                          EventType.PARTNER_ATTEMPT, person)
            return

        # ***** Forman pareja *****
        person.partner    = candidate
        candidate.partner = person
        self._unregister_single(person)
        self._unregister_single(candidate)
        self.total_couples_formed += 1

        # Se agenda el primer chequeo de ruptura
        self.schedule(evt.time + self.rng.exponential(BREAKUP_CHECK_LAM),
                      EventType.BREAKUP, person)

        # Solo la mujer agenda el proceso de embarazo
        woman = person if person.sex == "F" else candidate
        self.schedule(evt.time + self.rng.exponential(PREGNANCY_CHECK_LAM),
                      EventType.PREGNANCY, woman)

    def handle_breakup(self, evt: Event) -> None:
        # Procesa un evento de posible ruptura de pareja
        person = evt.person
        if not person.alive or person.partner is None:
            return
        if self.rng.bernoulli(BREAKUP_PROB):
            self._dissolve_couple(person, evt.time)
            self.total_breakups += 1
        else:
            # Si no se rompe, se programa otro chequeo en el futuro
            self.schedule(evt.time + self.rng.exponential(BREAKUP_CHECK_LAM),
                          EventType.BREAKUP, person)

    def handle_pregnancy(self, evt: Event) -> None:
        """
        Verifica que la mujer no esté ya embarazada antes de intentar
        un nuevo embarazo. Esto evita embarazos concurrentes.
        """
        woman = evt.person
        if not woman.alive or woman.sex != "F":
            return
        if woman.partner is None or not woman.partner.alive:
            return
        if woman.in_grief or woman.pregnant:
            self.schedule(evt.time + self.rng.exponential(PREGNANCY_CHECK_LAM),
                          EventType.PREGNANCY, woman)
            return

        # Actualiza la edad actual al instante del chequeo
        woman.age = woman.age_at(evt.time)
        pp        = woman.pregnancy_prob()

        # No puede embarazarse si ya alcanzó su cuota o si la pareja también
        # alcanzó la suya
        if (woman.children >= woman.max_children
                or woman.partner.children >= woman.partner.max_children
                or pp is None):
            return

        # Si se cumple la probabilidad, se marca el embarazo y se agenda el parto
        if self.rng.bernoulli(pp):
            woman.pregnant = True
            self.schedule(evt.time + PREGNANCY_DURATION,
                          EventType.BIRTH, woman)

        # Si no quedó embarazada, se reintenta más adelante
        if not woman.pregnant:
            self.schedule(evt.time + self.rng.exponential(PREGNANCY_CHECK_LAM),
                          EventType.PREGNANCY, woman)

    def handle_birth(self, evt: Event) -> None:
        """Nacimiento: crea bebés, incrementa contadores, agenda su destino."""
        mother = evt.person
        if not mother.alive:
            mother.pregnant = False
            return

        # El embarazo termina al ocurrir el parto
        mother.pregnant = False

        # Número base de bebés según distribución de parto múltiple
        n_babies = self.rng.discrete_choice(MULTIPLE_BIRTH_VALUES,
                                             MULTIPLE_BIRTH_PROBS)
        father        = mother.partner
        mother_quota  = mother.max_children - mother.children
        father_quota  = (father.max_children - father.children
                         if father and father.alive else 999)

        # Ajuste por cupos restantes de ambos padres
        n_babies = min(n_babies, max(0, min(mother_quota, father_quota)))

        # Se crean todos los bebés del parto
        for _ in range(n_babies):
            sex  = "F" if self.rng.bernoulli(0.5) else "M"
            baby = Person(
                pid          = self._new_pid(),
                sex          = sex,
                age          = 0.0,
                birth_time   = evt.time,
                # Cada bebé recibe su propio máximo de hijos deseados
                max_children = sample_max_children(self.rng),
            )
            self.population.append(baby)
            self.total_births += 1
            # Desde su nacimiento se agenda su destino vital
            self._schedule_fate(baby, evt.time)

        # Se actualizan los hijos de ambos padres
        mother.children += n_babies
        if father and father.alive:
            father.children += n_babies

        # Si la pareja sigue activa y aún puede tener más hijos,
        # se agenda otro chequeo de embarazo
        if (mother.alive and mother.partner and mother.partner.alive
                and mother.children < min(mother.max_children,
                                          mother.partner.max_children)):
            self.schedule(evt.time + self.rng.exponential(PREGNANCY_CHECK_LAM),
                          EventType.PREGNANCY, mother)

    # ****** Snapshots *****

    def _snapshot(self) -> None:
        # Cuenta cuántas personas siguen vivas en el instante del snapshot
        alive = sum(1 for p in self.population if p.alive)
        self.population_snapshot.append((round(self.clock, 1), alive))

    # ***** Loop principal *****

    def run(self) -> None:
        print(f"\n{'═'*55}")
        print(f"  Simulación iniciada  |  horizonte = {self.horizon:.0f} años")
        print(f"  Población inicial    |  {len(self.population)} personas")
        print(f"{'═'*55}\n")

        # Agenda el destino vital inicial de toda la población existente
        for person in self.population:
            self._schedule_fate(person, entry_time=0.0)

        # Intenta incorporar a todos los individuos al pool de solteros
        for person in self.population:
            self._try_enter_partner_search(person, 0.0)

        # Toma snapshot inicial
        self._snapshot()

        processed = 0
        # Procesa eventos hasta que se vacíe la FEL o se supere el horizonte
        while self._fel:
            evt = heapq.heappop(self._fel)
            if evt.time > self.horizon:
                break
            self.clock = evt.time

            # Toma snapshot cada 10 años
            if self.clock >= self._next_snapshot:
                self._snapshot()
                self._next_snapshot += 10.0

            # Limpieza periódica de entradas obsoletas en los pools de solteros
            self._events_since_cleanup += 1
            if self._events_since_cleanup >= self._CLEANUP_INTERVAL:
                self._cleanup_singles()
                self._events_since_cleanup = 0

            # Despacho del evento según su tipo
            match evt.event_type:
                case EventType.DEATH:            self.handle_death(evt)
                case EventType.AGE_TRANSITION:   self.handle_age_transition(evt)
                case EventType.GRIEF_END:        self.handle_grief_end(evt)
                case EventType.PARTNER_ATTEMPT:  self.handle_partner_attempt(evt)
                case EventType.BREAKUP:          self.handle_breakup(evt)
                case EventType.PREGNANCY:        self.handle_pregnancy(evt)
                case EventType.BIRTH:            self.handle_birth(evt)

            processed += 1

        # Se fija el reloj al horizonte final y se toma snapshot final
        self.clock = self.horizon
        self._snapshot()
        print(f"  Eventos procesados : {processed:,}")
        print(f"  Reloj final        : {self.clock:.1f} años\n")

    # ***** Reporte *****

    def report(self) -> None:
        # Personas vivas al final
        alive   = [p for p in self.population if p.alive]
        alive_f = sum(1 for p in alive if p.sex == "F")
        alive_m = sum(1 for p in alive if p.sex == "M")
        # Personas nacidas durante la simulación
        born    = [p for p in self.population if p.birth_time > 0]
        sep     = "─" * 55

        print(f"\n{sep}")
        print("  REPORTE FINAL — Poblado en Evolución")
        print(sep)
        print(f"  Horizonte           : {self.horizon:.0f} años")
        print(f"  Población inicial   : {len(self.population) - len(born)}")
        print(f"  Nacidos en sim.     : {len(born)}")
        print(f"  Población total     : {len(self.population)}")
        print(f"  Vivos al final      : {len(alive)}  (F={alive_f}, M={alive_m})")
        print(f"  Fallecidos totales  : {self.total_deaths}")
        print(f"  Parejas formadas    : {self.total_couples_formed}")
        print(f"  Rupturas            : {self.total_breakups}")

        print(f"\n  Muertes por sexo:")
        for sex, n in sorted(self.deaths_by_sex.items()):
            print(f"    {'Mujeres' if sex=='F' else 'Hombres':10s}: {n}")

        print(f"\n  Muertes por rango de edad:")
        for rango in ["0-12", "12-45", "45-76", "76-125"]:
            n   = self.deaths_by_age_range.get(rango, 0)
            bar = "█" * min(n // 2, 40)
            print(f"    {rango:>8s} → {n:4d}  {bar}")

        print(f"\n  Evolución de la población:")
        print(f"  {'Año':>5}  {'Vivos':>6}")
        print(f"  {'─'*14}")
        init = max(1, len(self.population) - len(born))
        for year, n in self.population_snapshot:
            bar = "█" * (n * 30 // init)
            print(f"  {year:>5.0f}  {n:>6}  {bar}")
        print(sep)