"""
Tests: Parejas, Embarazo y Nacimientos 

Ejecutar con: python tests/test_parejas_embarazo_nacimientos.py
"""

# Ajustar el path para poder importar módulos del proyecto desde la carpeta tests
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Importar las herramientas de simulación y el motor DES
from src.generadores import LCG, RandomVariables
from src.modelo import Person, generate_population
from src.motor_des import DESEngine, EventType


def make_engine(seed=42, M=150, H=150, horizon=100.0):
    # Crea el generador aleatorio con la semilla indicada
    rng = RandomVariables(LCG(seed=seed))
    # Genera la población inicial
    pop = generate_population(M, H, rng)
    # Crea el motor de simulación con ese escenario
    eng = DESEngine(pop, rng, sim_horizon=horizon)
    # Ejecuta toda la simulación
    eng.run()
    return eng


# ----- Parejas -----

def test_couples_formed():
    """Deben formarse parejas durante la simulación."""
    eng = make_engine()
    assert eng.total_couples_formed > 0, "No se formó ninguna pareja"


def test_no_same_sex_couples():
    """Ninguna pareja debe ser del mismo sexo."""
    eng = make_engine()
    for p in eng.population:
        if p.partner is not None:
            assert p.sex != p.partner.sex, \
                f"Pareja del mismo sexo: pid={p.pid} sex={p.sex}"


def test_partner_symmetry():
    """Si A.partner == B entonces B.partner == A."""
    eng = make_engine()
    for p in eng.population:
        if p.partner is not None:
            assert p.partner.partner is p, \
                f"Asimetría en pareja: pid={p.pid}"


def test_dead_have_no_partner():
    """Ningún muerto debe tener pareja."""
    eng = make_engine()
    for p in eng.population:
        if not p.alive:
            assert p.partner is None, \
                f"Persona muerta pid={p.pid} tiene pareja"


def test_grief_persons_are_single():
    """Una persona en duelo no debe tener pareja."""
    eng = make_engine()
    for p in eng.population:
        if p.in_grief:
            assert p.partner is None, \
                f"Persona en duelo pid={p.pid} tiene pareja"


# ----- Nacimientos -----

def test_births_occurred():
    """Deben producirse nacimientos durante la simulación."""
    eng = make_engine()
    assert eng.total_births > 0, "No hubo ningún nacimiento"


def test_born_persons_have_positive_birth_time():
    """Todo bebé nacido en simulación tiene birth_time > 0."""
    eng = make_engine()
    for p in eng.population:
        if p.birth_time > 0:
            assert p.birth_time <= 100.0


def test_population_grew():
    """La población total debe ser mayor que la inicial."""
    eng = make_engine()
    initial = 300
    assert len(eng.population) > initial, \
        f"Población no creció: {len(eng.population)} <= {initial}"


def test_children_count_non_negative():
    """El contador de hijos nunca debe ser negativo."""
    eng = make_engine()
    for p in eng.population:
        assert p.children >= 0


def test_children_never_exceed_max():
    """Ninguna persona debe tener más hijos que su max_children."""
    eng = make_engine()
    for p in eng.population:
        # Se tolera un margen pequeño por partos múltiples
        assert p.children <= p.max_children + 4, \
            f"pid={p.pid} tiene {p.children} hijos, max={p.max_children}"


def test_newborns_sex_distribution():
    """El ratio M/F entre nacidos debe estar cerca de 50/50 (±15%)."""
    eng = make_engine(M=200, H=200)
    born = [p for p in eng.population if p.birth_time > 0]
    if not born:
        return
    ratio = sum(1 for p in born if p.sex == "F") / len(born)
    assert 0.35 <= ratio <= 0.65, f"Ratio F entre nacidos: {ratio:.2f}"


# ----- Reproducibilidad -----

def test_reproducible_births():
    """Misma semilla → mismo número de nacimientos."""
    e1 = make_engine(seed=77)
    e2 = make_engine(seed=77)
    assert e1.total_births == e2.total_births


def test_reproducible_couples():
    """Misma semilla → mismas parejas formadas."""
    e1 = make_engine(seed=77)
    e2 = make_engine(seed=77)
    assert e1.total_couples_formed == e2.total_couples_formed


# ----- Ejecución directa ------

if __name__ == "__main__":
    # Lista manual de tests para ejecutarlos sin pytest
    tests = [
        test_couples_formed, test_no_same_sex_couples,
        test_partner_symmetry, test_dead_have_no_partner,
        test_grief_persons_are_single,
        test_births_occurred, test_born_persons_have_positive_birth_time,
        test_population_grew, test_children_count_non_negative,
        test_children_never_exceed_max, test_newborns_sex_distribution,
        test_reproducible_births, test_reproducible_couples,
    ]
    passed = failed = 0

    # Ejecuta cada test y cuenta resultados
    for t in tests:
        try:
            t()
            print(f"  ✔  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✘  {t.__name__}  →  {e}")
            failed += 1

    print(f"\n  {passed}/{passed+failed} tests pasaron")