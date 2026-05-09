"""
Tests: Modelo y Motor DES
==========================
Verifica Person, generate_population y DESEngine.
Ejecutar con: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generadores import LCG, RandomVariables
from src.modelo import Person, generate_population, DEATH_PROB
from src.motor_des import DESEngine, EventType


def make_rng(seed: int = 42) -> RandomVariables:
    return RandomVariables(LCG(seed=seed))


# ─── Person ─────────────────────────────────────────────────────────────────

def test_person_death_prob_in_range():
    """death_prob debe estar en [0, 1] para toda edad válida."""
    for sex in ("M", "F"):
        for lo, hi, _ in DEATH_PROB[sex]:
            p = Person(pid=0, sex=sex, age=(lo + hi) / 2)
            dp = p.death_prob()
            assert dp is not None and 0 <= dp <= 1


def test_person_pregnancy_only_female():
    male = Person(pid=0, sex="M", age=25)
    assert male.pregnancy_prob() is None


def test_person_is_single():
    p = Person(pid=0, sex="F", age=25)
    assert p.is_single()
    p.in_grief = True
    assert not p.is_single()


def test_person_age_range_label():
    p = Person(pid=0, sex="M", age=30)
    assert p.age_range_label() == "12-45"


# ─── Población inicial ───────────────────────────────────────────────────────

def test_population_size():
    rng = make_rng()
    pop = generate_population(30, 20, rng)
    assert len(pop) == 50
    assert sum(1 for p in pop if p.sex == "F") == 30
    assert sum(1 for p in pop if p.sex == "M") == 20


def test_population_age_range():
    rng = make_rng()
    pop = generate_population(50, 50, rng)
    assert all(0.0 <= p.age <= 100.0 for p in pop)


def test_population_unique_pids():
    rng = make_rng()
    pop = generate_population(50, 50, rng)
    pids = [p.pid for p in pop]
    assert len(pids) == len(set(pids))


def test_population_max_children_valid():
    rng = make_rng()
    pop = generate_population(100, 100, rng)
    assert all(1 <= p.max_children <= 6 for p in pop)


# ─── Motor DES ───────────────────────────────────────────────────────────────

def _run_small(seed=42, n=100, horizon=100.0):
    rng = make_rng(seed)
    pop = generate_population(n // 2, n // 2, rng)
    engine = DESEngine(pop, rng, sim_horizon=horizon)
    engine.run()
    return engine


def test_engine_clock_at_horizon():
    engine = _run_small()
    assert engine.clock == 100.0


def test_engine_death_count_positive():
    engine = _run_small()
    assert engine.total_deaths > 0


def test_engine_no_alive_contradictions():
    """Una persona muerta no puede tener pareja viva."""
    engine = _run_small()
    for p in engine.population:
        if not p.alive:
            assert p.partner is None or not p.partner.alive


def test_engine_deaths_never_exceed_population():
    engine = _run_small()
    assert engine.total_deaths <= len(engine.population)


def test_engine_snapshots_exist():
    """Deben existir snapshots a lo largo de la simulación."""
    engine = _run_small()
    assert len(engine.population_snapshot) >= 2


def test_engine_reproducible():
    """Misma semilla → mismos resultados."""
    e1 = _run_small(seed=7)
    e2 = _run_small(seed=7)
    assert e1.total_deaths == e2.total_deaths


def test_engine_different_seeds_differ():
    e1 = _run_small(seed=1)
    e2 = _run_small(seed=2)
    # Con semillas distintas los resultados casi nunca coinciden
    assert e1.total_deaths != e2.total_deaths


# ─── Ejecución directa ───────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_person_death_prob_in_range, test_person_pregnancy_only_female,
        test_person_is_single, test_person_age_range_label,
        test_population_size, test_population_age_range,
        test_population_unique_pids, test_population_max_children_valid,
        test_engine_clock_at_horizon, test_engine_death_count_positive,
        test_engine_no_alive_contradictions, test_engine_deaths_never_exceed_population,
        test_engine_snapshots_exist, test_engine_reproducible,
        test_engine_different_seeds_differ,
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