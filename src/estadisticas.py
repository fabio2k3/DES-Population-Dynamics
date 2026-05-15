"""
Módulo: Recolector de Estadísticas

Inyecta hooks en el DESEngine para recopilar métricas detalladas
sin mezclar lógica de análisis con lógica de simulación.

Métricas recopiladas:
  - Población total, mujeres y hombres vivos en cada snapshot
  - Nacimientos y muertes acumuladas por década
  - Ratio F/M a lo largo del tiempo
  - Distribución de edades en t=0, t=50 y t=100
  - Parejas activas por snapshot
  - Resumen por corrida (para análisis multi-corrida)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Solo se importa DESEngine para tipado estático.
# TYPE_CHECKING evita importar el motor en tiempo de ejecución.
if TYPE_CHECKING:
    from src.motor_des import DESEngine


# ====================
# SNAPSHOT DETALLADO =
# ====================

@dataclass
class Snapshot:
    # Año del snapshot dentro del horizonte de simulación
    year          : float
    # Total de personas vivas en ese instante
    total_alive   : int
    # Cantidad de mujeres vivas
    alive_f       : int
    # Cantidad de hombres vivos
    alive_m       : int
    # Parejas activas en ese snapshot
    couples_active: int
    # Nacimientos acumulados hasta ese año
    births_accum  : int
    # Muertes acumuladas hasta ese año
    deaths_accum  : int

    @property
    def ratio_fm(self) -> float:
        """Ratio F / total entre vivos. 0.5 = equilibrio."""
        return self.alive_f / self.total_alive if self.total_alive else 0.0


# ========================
# RESUMEN DE UNA CORRIDA =
# ========================

@dataclass
class RunSummary:
    # Semilla utilizada para reproducibilidad
    seed              : int
    # Población inicial de la simulación
    initial_pop       : int
    # Personas vivas al final de la corrida
    final_alive       : int
    # Total de nacimientos observados
    total_births      : int
    # Total de muertes observadas
    total_deaths      : int
    # Total de parejas formadas durante la simulación
    couples_formed    : int
    # Total de rupturas de pareja
    total_breakups    : int
    # Snapshots completos de la corrida
    snapshots         : list[Snapshot] = field(default_factory=list)
    # Distribución de edades en t = 0
    age_dist_t0       : list[float]    = field(default_factory=list)
    # Distribución de edades en t = 50
    age_dist_t50      : list[float]    = field(default_factory=list)
    # Distribución de edades en t = 100
    age_dist_t100     : list[float]    = field(default_factory=list)
    # Nacimientos por década
    births_per_decade : list[int]      = field(default_factory=list)
    # Muertes por década
    deaths_per_decade : list[int]      = field(default_factory=list)

    @property
    def growth_rate(self) -> float:
        """Cambio porcentual de población respecto al inicio."""
        return (self.final_alive - self.initial_pop) / self.initial_pop * 100


# ===========
# COLECTOR  =
# ===========

class StatsCollector:
    """
    Se conecta a un DESEngine ya ejecutado y extrae todas las métricas.

    Uso:
        engine.run()
        collector = StatsCollector(engine, seed=42)
        summary   = collector.collect()
    """

    # Años exactos en los que se construyen snapshots
    SNAPSHOT_YEARS = list(range(0, 101, 10))   # cada 10 años

    def __init__(self, engine: "DESEngine", seed: int = 0):
        # Referencia al motor de simulación ya ejecutado
        self.engine = engine
        # Semilla asociada a esta corrida
        self.seed   = seed

    def collect(self) -> RunSummary:
        # Acceso directo al motor y a la población simulada
        eng = self.engine
        pop = eng.population

        # Se construyen snapshots exactos usando la información ya registrada
        # por el motor. Esto evita duplicar lógica o mantener cálculos muertos.
        snapshots = self._build_snapshots_exact()

        # ***** Distribuciones de edad *****
        # Edad de las personas que ya existían al inicio de la simulación
        age_t0   = [p.age_at(0)           for p in pop if p.birth_time <= 0]

        # Edad en t=50 para quienes ya habían nacido y cumplen la condición
        age_t50  = [p.age_at(50)          for p in pop
                    if p.birth_time <= 50 and (p.alive or p.age >= p.age_at(50))]

        # Edad en el horizonte final para quienes siguen vivos
        age_t100 = [p.age_at(eng.horizon) for p in pop if p.alive]

        # ***** Nacimientos por década *****
        # Se cuentan solo nacimientos posteriores al inicio
        births_per_decade = self._count_per_decade(
            [p.birth_time for p in pop if p.birth_time > 0]
        )

        return RunSummary(
            seed              = self.seed,
            initial_pop       = sum(1 for p in pop if p.birth_time <= 0),
            final_alive       = sum(1 for p in pop if p.alive),
            total_births      = eng.total_births,
            total_deaths      = eng.total_deaths,
            couples_formed    = eng.total_couples_formed,
            total_breakups    = eng.total_breakups,
            snapshots         = snapshots,
            age_dist_t0       = age_t0,
            age_dist_t50      = age_t50,
            age_dist_t100     = age_t100,
            births_per_decade = births_per_decade,
            deaths_per_decade = self._deaths_per_decade(),
        )

    def _build_snapshots_exact(self) -> list[Snapshot]:
        """
        Construye snapshots usando directamente los datos del engine.

        El motor ya guarda pares (año, n_vivos) en population_snapshot,
        por lo que aquí solo se reordena y se completa la información.

        couples_active se estima contando mujeres vivas con pareja.
        """
        eng  = self.engine
        pop  = eng.population
        snaps: list[Snapshot] = []

        # Convertimos la lista de snapshots del motor en un mapa por año
        snap_map = {}
        for year, n in eng.population_snapshot:
            snap_map.setdefault(int(round(year)), n)

        # Se construye un snapshot para cada año objetivo
        for year in self.SNAPSHOT_YEARS:
            n_total = snap_map.get(year, 0)

            # Conteo base de mujeres vivas en ese año
            alive_f = sum(
                1 for p in pop
                if p.sex == "F"
                and p.birth_time <= year
                and (p.alive or p.age > (year - p.birth_time))
            )

            # Conteo base de hombres vivos en ese año
            alive_m = sum(
                1 for p in pop
                if p.sex == "M"
                and p.birth_time <= year
                and (p.alive or p.age > (year - p.birth_time))
            )

            # Normalización para ajustar el conteo al valor registrado por el motor
            total_fm = alive_f + alive_m
            if total_fm > 0 and n_total > 0:
                scale   = n_total / total_fm
                alive_f = round(alive_f * scale)
                alive_m = n_total - alive_f

            # Conteo aproximado de parejas activas al final de la simulación
            # usando solo el lado femenino para evitar doble conteo.
            couples_final = sum(
                1 for p in pop
                if p.sex == "F" and p.alive and p.partner is not None
            )

            # Ajuste proporcional del número de parejas según población viva
            final_alive = sum(1 for p in pop if p.alive)
            if final_alive > 0 and n_total > 0:
                couples_at_year = round(couples_final * n_total / final_alive)
            else:
                couples_at_year = couples_final if year == eng.horizon else 0

            # Se crea el snapshot final con todos los indicadores requeridos
            snaps.append(Snapshot(
                year           = year,
                total_alive    = n_total,
                alive_f        = max(0, alive_f),
                alive_m        = max(0, alive_m),
                couples_active = couples_at_year,
                births_accum   = eng.total_births,
                deaths_accum   = eng.total_deaths,
            ))

        return snaps

    def _count_per_decade(self, times: list[float]) -> list[int]:
        """
        Cuenta cuántos eventos caen en cada década:
        [0-10), [10-20), ..., [90-100).

        Se limita a eventos dentro del horizonte 0 < t <= 100.
        """
        counts = [0] * 10
        for t in times:
            if 0 < t <= 100:
                idx = min(int(t // 10), 9)
                counts[idx] += 1
        return counts

    def _deaths_per_decade(self) -> list[int]:
        """Muertes por década usando la edad real al fallecer."""
        counts = [0] * 10
        for p in self.engine.population:
            if not p.alive:
                # Si la persona murió, el instante de muerte se reconstruye
                # como birth_time + age, dado que age se actualiza al morir.
                t_death = p.birth_time + p.age
                if 0 < t_death <= 100:
                    idx = min(int(t_death // 10), 9)
                    counts[idx] += 1
        return counts


# =========================
# ANÁLISIS MULTI-CORRIDA  =
# =========================
@dataclass
class MultiRunAnalysis:
    """Estadísticas agregadas sobre N corridas independientes."""
    # Número total de simulaciones analizadas
    n_runs         : int
    # Lista de resúmenes individuales
    summaries      : list[RunSummary]

    def _values(self, attr: str) -> list:
        # Extrae el valor de un atributo de cada corrida
        return [getattr(s, attr) for s in self.summaries]

    def mean(self, attr: str) -> float:
        # Media aritmética del atributo seleccionado
        vals = self._values(attr)
        return sum(vals) / len(vals)

    def std(self, attr: str) -> float:
        # Desviación estándar poblacional del atributo seleccionado
        vals  = self._values(attr)
        m     = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    def ci_95(self, attr: str) -> tuple[float, float]:
        """Intervalo de confianza del 95% (aproximación normal)."""
        m   = self.mean(attr)
        s   = self.std(attr)
        z   = 1.96
        n   = self.n_runs
        margin = z * s / (n ** 0.5)
        return (m - margin, m + margin)

    def pop_trajectory(self) -> tuple[list[float], list[float], list[float]]:
        """
        Devuelve:
            years      -> años de los snapshots
            mean_alive -> población media viva en cada snapshot
            std_alive  -> desviación estándar en cada snapshot

        Esto sirve para graficar la trayectoria media con su variabilidad.
        """
        years = [s.year for s in self.summaries[0].snapshots]
        by_year: list[list[int]] = [[] for _ in years]

        # Agrupa la población viva de todas las corridas por año
        for summary in self.summaries:
            for i, snap in enumerate(summary.snapshots):
                by_year[i].append(snap.total_alive)

        # Media por punto temporal
        means = [sum(v) / len(v) for v in by_year]

        # Desviación estándar por punto temporal
        stds  = [
            (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
            for v, m in zip(by_year, means)
        ]
        return years, means, stds

    def print_summary(self) -> None:
        # Separador visual para imprimir el reporte en consola
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  ANÁLISIS MULTI-CORRIDA  ({self.n_runs} corridas)")
        print(sep)

        # Métricas principales que se resumen con media, desviación e IC95%
        metrics = [
            ("Población final viva",   "final_alive"),
            ("Nacimientos totales",    "total_births"),
            ("Muertes totales",        "total_deaths"),
            ("Parejas formadas",       "couples_formed"),
            ("Rupturas",               "total_breakups"),
        ]
        for label, attr in metrics:
            m  = self.mean(attr)
            sd = self.std(attr)
            lo, hi = self.ci_95(attr)
            print(f"  {label:28s}: {m:7.1f} ± {sd:5.1f}  "
                  f"IC95% [{lo:6.1f}, {hi:6.1f}]")
        print(sep)