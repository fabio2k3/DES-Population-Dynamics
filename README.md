# Poblado en Evolución — Simulación de Eventos Discretos

## El Problema

Se simula la evolución demográfica de un poblado durante **100 años**.
Cada individuo tiene sexo, edad y estado civil, y puede experimentar los
siguientes eventos a lo largo de su vida:

- Buscar pareja y formar una relación
- Reproducirse (con posibilidad de partos múltiples)
- Separarse o enviudar y atravesar un período de duelo
- Morir según probabilidades diferenciadas por sexo y rango de edad

El sistema no itera año a año — avanza directamente de evento en evento,
lo que permite modelar con precisión procesos como duelos de 3 meses o
embarazos de 9 meses sin errores de discretización.

---

## Solución: Motor DES con FEL

Se implementó un motor de **Simulación de Eventos Discretos (DES)** puro:

- **Generador LCG** propio (parámetros glibc, período 2³¹) sin usar
  `random.random()`. Todas las distribuciones usan transformada inversa.
- **Future Event List (FEL)** implementada con `heapq` — extracción e
  inserción en O(log n).
- **7 tipos de evento**: `DEATH`, `AGE_TRANSITION`, `GRIEF_END`,
  `PARTNER_ATTEMPT`, `BREAKUP`, `PREGNANCY`, `BIRTH`.
- Las probabilidades del enunciado (mortalidad, embarazo, pareja, duelo)
  se consultan por rango de edad en cada evento.
- Análisis multi-corrida con media, desviación estándar e IC 95%.

```
src/
├── generadores.py   # LCG + transformada inversa
├── modelo.py        # Tablas del enunciado + clase Person
├── motor_des.py     # FEL, eventos, lógica de simulación
├── estadisticas.py  # Snapshots, métricas, análisis multi-corrida
└── visualizacion.py # Gráficas con matplotlib
```

---

## Cómo Ejecutar

### Corrida única (parámetros por defecto)
```bash
python main.py
```

### Corrida única con parámetros personalizados
```bash
python main.py --seed 7 --m 300 --h 300 --years 100
```

### Múltiples corridas con análisis estadístico
```bash
python main.py --runs 10
```

### Sin generar gráficas
```bash
python main.py --runs 10 --no-plots
```

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--seed`  | Semilla base del LCG | 42 |
| `--m`     | Número de mujeres iniciales | 200 |
| `--h`     | Número de hombres iniciales | 200 |
| `--years` | Horizonte de simulación (años) | 100.0 |
| `--runs`  | Número de corridas independientes | 1 |
| `--no-plots` | No generar gráficas | False |

Las gráficas se guardan en `resultados/`.

---

## Tests

### Correr todos los tests de una vez
```bash
python tests/test_generadores.py
python tests/test_motor.py
python tests/test_parejas_embarazo_nacimientos.py
python tests/test_estadisticas.py
```

### Test de generadores (LCG, Uniforme, Exponencial, Bernoulli)
```bash
python tests/test_generadores.py
```
Verifica propiedades estadísticas: medias, rangos, reproducibilidad y
validación de argumentos inválidos. **13 tests.**

### Test del motor y modelo (Person, DESEngine, población)
```bash
python tests/test_motor.py
```
Verifica invariantes del motor: reloj al horizonte, muertes positivas,
reproducibilidad por semilla, consistencia de la población. **15 tests.**

### Test de parejas y nacimientos
```bash
python tests/test_parejas_embarazo_nacimientos.py
```
Verifica lógica de emparejamiento: simetría de parejas, muertos sin
pareja, distribución de sexo al nacer, reproducibilidad. **13 tests.**

### Test de estadísticas y multi-corrida
```bash
python tests/test_estadisticas.py
```
Verifica snapshots, ratios F/M, décadas, IC 95% y trayectoria
multi-corrida. **21 tests.**

---

## Dependencias

```bash
pip install matplotlib numpy
```

> **Nota:** `matplotlib` y `numpy` se usan **exclusivamente para
> visualización** (`src/visualizacion.py`). Toda la lógica de simulación,
> los generadores de variables aleatorias (LCG + transformada inversa),
> el motor DES y el análisis estadístico están implementados desde cero
> sin dependencias externas, cumpliendo con los requisitos del curso.

Python 3.10+ requerido (se usa `match` para el dispatch de eventos).

---
