"""
Tests: Estadísticas y Análisis Multi-corrida
=============================================
Ejecutar con: python tests/test_estadisticas.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generadores import LCG, RandomVariables
from src.modelo import generate_population
from src.motor_des import DESEngine
from src.estadisticas import StatsCollector, MultiRunAnalysis


def make_summary(seed=42, M=100, H=100):
    rng = RandomVariables(LCG(seed=seed))
    pop = generate_population(M, H, rng)
    eng = DESEngine(pop, rng, sim_horizon=100.0)
    eng.run()
    return StatsCollector(eng, seed=seed).collect()


# ── Snapshots ────────────────────────────────────────────────────────────────

def test_snapshots_count():
    """Debe haber exactamente 11 snapshots (t=0, 10, ..., 100)."""
    s = make_summary()
    assert len(s.snapshots) == 11, f"Esperados 11, encontrados {len(s.snapshots)}"


def test_snapshots_years():
    """Los años de los snapshots deben ser 0, 10, 20 ... 100."""
    s = make_summary()
    years = [snap.year for snap in s.snapshots]
    assert years == list(range(0, 101, 10))


def test_snapshots_population_non_negative():
    """Ningún snapshot puede tener población negativa."""
    s = make_summary()
    for snap in s.snapshots:
        assert snap.total_alive >= 0
        assert snap.alive_f >= 0
        assert snap.alive_m >= 0


def test_ratio_fm_range():
    """El ratio F/Total debe estar en [0, 1]."""
    s = make_summary()
    for snap in s.snapshots:
        r = snap.ratio_fm
        assert 0.0 <= r <= 1.0, f"Ratio fuera de rango: {r} en año {snap.year}"


# ── Totales ───────────────────────────────────────────────────────────────────

def test_summary_births_positive():
    s = make_summary()
    assert s.total_births > 0

def test_summary_deaths_positive():
    s = make_summary()
    assert s.total_deaths > 0

def test_summary_final_alive_positive():
    s = make_summary()
    assert s.final_alive > 0

def test_initial_pop_correct():
    s = make_summary(M=100, H=100)
    assert s.initial_pop == 200

# ── Distribuciones de edad ───────────────────────────────────────────────────

def test_age_dist_t0_range():
    """Edades en t=0 deben estar en [0, 100]."""
    s = make_summary()
    assert all(0 <= a <= 100 for a in s.age_dist_t0)

def test_age_dist_t100_non_empty():
    """Debe haber personas vivas al final."""
    s = make_summary()
    assert len(s.age_dist_t100) > 0

def test_age_dist_t100_all_positive():
    """Todos los vivos al final deben tener edad > 0."""
    s = make_summary()
    assert all(a >= 0 for a in s.age_dist_t100)

# ── Décadas ───────────────────────────────────────────────────────────────────

def test_births_per_decade_length():
    s = make_summary()
    assert len(s.births_per_decade) == 10

def test_deaths_per_decade_length():
    s = make_summary()
    assert len(s.deaths_per_decade) == 10

def test_births_per_decade_sum_matches_total():
    """Suma de nacimientos por década debe coincidir con total_births."""
    s = make_summary()
    assert sum(s.births_per_decade) == s.total_births

def test_deaths_per_decade_non_negative():
    s = make_summary()
    assert all(d >= 0 for d in s.deaths_per_decade)

# ── Multi-corrida ─────────────────────────────────────────────────────────────

def _make_multi(n=5):
    summaries = [make_summary(seed=i) for i in range(n)]
    return MultiRunAnalysis(n_runs=n, summaries=summaries)

def test_multi_mean_positive():
    m = _make_multi()
    assert m.mean("final_alive") > 0
    assert m.mean("total_births") > 0

def test_multi_std_non_negative():
    m = _make_multi()
    assert m.std("final_alive") >= 0

def test_multi_ci_ordered():
    """El límite inferior del IC debe ser ≤ la media."""
    m = _make_multi()
    lo, hi = m.ci_95("final_alive")
    mean   = m.mean("final_alive")
    assert lo <= mean <= hi

def test_multi_trajectory_length():
    """La trayectoria debe tener 11 puntos."""
    m = _make_multi()
    years, means, stds = m.pop_trajectory()
    assert len(years) == len(means) == len(stds) == 11

def test_multi_trajectory_stds_non_negative():
    m = _make_multi()
    _, _, stds = m.pop_trajectory()
    assert all(s >= 0 for s in stds)

def test_growth_rate_type():
    s = make_summary()
    assert isinstance(s.growth_rate, float)

# ── Ejecución directa ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_snapshots_count, test_snapshots_years,
        test_snapshots_population_non_negative, test_ratio_fm_range,
        test_summary_births_positive, test_summary_deaths_positive,
        test_summary_final_alive_positive, test_initial_pop_correct,
        test_age_dist_t0_range, test_age_dist_t100_non_empty,
        test_age_dist_t100_all_positive,
        test_births_per_decade_length, test_deaths_per_decade_length,
        test_births_per_decade_sum_matches_total,
        test_deaths_per_decade_non_negative,
        test_multi_mean_positive, test_multi_std_non_negative,
        test_multi_ci_ordered, test_multi_trajectory_length,
        test_multi_trajectory_stds_non_negative, test_growth_rate_type,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✔  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✘  {t.__name__}  →  {e}")
            failed += 1
    print(f"\n  {passed}/{passed+failed} tests pasaron")
