"""
Tests: Generadores de Variables Aleatorias
==========================================
Verifica propiedades estadísticas básicas de LCG y RandomVariables.
Ejecutar con: python -m pytest tests/ -v
"""

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generadores import LCG, RandomVariables

N = 20_000   # muestras para tests estadísticos


def make_rng(seed: int = 0) -> RandomVariables:
    return RandomVariables(LCG(seed=seed))


# ─── LCG ────────────────────────────────────────────────────────────────────

def test_lcg_range():
    """Todos los valores deben estar en (0, 1)."""
    lcg = LCG(seed=7)
    vals = [lcg.next_uniform() for _ in range(N)]
    assert all(0 < v < 1 for v in vals), "Valor fuera de (0, 1)"


def test_lcg_reproducible():
    """Misma semilla → misma secuencia."""
    a = [LCG(seed=99).next_uniform() for _ in range(20)]
    b = [LCG(seed=99).next_uniform() for _ in range(20)]
    assert a == b


def test_lcg_different_seeds():
    """Semillas distintas → secuencias distintas."""
    a = LCG(seed=1).next_uniform()
    b = LCG(seed=2).next_uniform()
    assert a != b


# ─── Uniforme ────────────────────────────────────────────────────────────────

def test_uniform_range():
    rng = make_rng()
    vals = [rng.uniform(3.0, 7.0) for _ in range(N)]
    assert all(3.0 <= v <= 7.0 for v in vals)


def test_uniform_mean():
    """E[U(a,b)] = (a+b)/2  ±  tolerancia."""
    a, b = 2.0, 10.0
    rng  = make_rng()
    mean = sum(rng.uniform(a, b) for _ in range(N)) / N
    assert abs(mean - (a + b) / 2) < 0.15, f"Media={mean:.4f}, esperada={(a+b)/2}"


def test_uniform_bad_args():
    rng = make_rng()
    try:
        rng.uniform(5.0, 3.0)
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass


# ─── Exponencial ────────────────────────────────────────────────────────────

def test_exponential_positive():
    rng  = make_rng()
    vals = [rng.exponential(2.0) for _ in range(N)]
    assert all(v > 0 for v in vals)


def test_exponential_mean():
    """E[Exp(λ)] = 1/λ  ±  tolerancia."""
    lam  = 3.0
    rng  = make_rng()
    mean = sum(rng.exponential(lam) for _ in range(N)) / N
    assert abs(mean - 1.0 / lam) < 0.05, f"Media={mean:.4f}, esperada={1/lam:.4f}"


def test_exponential_inverse_transform():
    """Verifica la fórmula: X = -(1/λ)·ln(U) produce valores > 0."""
    lcg = LCG(seed=42)
    for _ in range(1000):
        u = lcg.next_uniform()
        x = -(1.0 / 2.0) * math.log(u)
        assert x > 0


def test_exponential_bad_lambda():
    rng = make_rng()
    try:
        rng.exponential(-1.0)
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass


# ─── Bernoulli ───────────────────────────────────────────────────────────────

def test_bernoulli_freq():
    """Frecuencia de True debe aproximarse a p."""
    p   = 0.3
    rng = make_rng()
    freq = sum(rng.bernoulli(p) for _ in range(N)) / N
    assert abs(freq - p) < 0.02, f"Freq={freq:.4f}, esperada={p}"


# ─── Elección discreta ───────────────────────────────────────────────────────

def test_discrete_choice_coverage():
    """Todos los valores deben aparecer en N muestras."""
    rng    = make_rng()
    values = [1, 2, 3]
    probs  = [1/3, 1/3, 1/3]
    seen   = set(rng.discrete_choice(values, probs) for _ in range(500))
    assert seen == {1, 2, 3}


def test_discrete_choice_bad_probs():
    rng = make_rng()
    try:
        rng.discrete_choice([1, 2], [0.4, 0.4])
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass


# ─── Ejecución directa ───────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_lcg_range, test_lcg_reproducible, test_lcg_different_seeds,
        test_uniform_range, test_uniform_mean, test_uniform_bad_args,
        test_exponential_positive, test_exponential_mean,
        test_exponential_inverse_transform, test_exponential_bad_lambda,
        test_bernoulli_freq,
        test_discrete_choice_coverage, test_discrete_choice_bad_probs,
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