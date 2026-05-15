"""
Módulo: Generadores de Variables Aleatorias

Contenido:
  - LCG               : Generador Congruencial Lineal (fuente de aleatoriedad)
  - RandomVariables   : Uniforme, Exponencial, Bernoulli, Elección discreta

No depende de ningún otro módulo del proyecto.
"""

import math


# =====================================
# LCG — Linear Congruential Generator =
# =====================================
# Fórmula:  X_{n+1} = (a · X_n + c) mod m
# Parámetros glibc: período completo 2^31, buen comportamiento estadístico.

class LCG:
    """Generador Congruencial Lineal. Produce U ~ Uniforme(0, 1)."""

    # Módulo del generador
    _M: int = 2**31
    # Multiplicador
    _A: int = 1_103_515_245
    # Incremento
    _C: int = 12_345

    def __init__(self, seed: int = 42):
        # Validación de la semilla: debe ser entera y no negativa
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("La semilla debe ser un entero no negativo.")
        # Estado interno inicializado con la semilla reducida al módulo
        self._state = seed % self._M

    def next_int(self) -> int:
        """Entero en [0, m-1]."""
        # Actualiza el estado interno usando la recurrencia del LCG
        self._state = (self._A * self._state + self._C) % self._M
        return self._state

    def next_uniform(self) -> float:
        """U ~ Uniforme(0, 1). Evita los extremos exactos para que ln(U) esté definido."""
        # Se desplaza en 0.5 para evitar exactamente 0 o 1
        return (self.next_int() + 0.5) / self._M

    def __repr__(self) -> str:
        # Representación útil para depuración
        return f"LCG(state={self._state}, m={self._M}, a={self._A}, c={self._C})"


# ================================================
# RandomVariables — colección de distribuciones  =
# ================================================

class RandomVariables:
    """
    Generadores de variables aleatorias basados en un LCG compartido.
    Todas las distribuciones usan Transformada Inversa, como exige el curso.
    """

    def __init__(self, lcg: LCG):
        # Se reutiliza un LCG externo para mantener una sola fuente de aleatoriedad
        self._lcg = lcg

    def uniform(self, a: float = 0.0, b: float = 1.0) -> float:
        """
        X ~ Uniforme(a, b)
        F^{-1}(U) = a + (b - a) · U
        """
        # Verifica que el intervalo sea válido
        if a >= b:
            raise ValueError(f"Se requiere a < b. Recibido: a={a}, b={b}.")
        # Transformación inversa de la uniforme estándar
        return a + (b - a) * self._lcg.next_uniform()

    def exponential(self, lam: float) -> float:
        """
        X ~ Exponencial(λ)
        F^{-1}(U) = -(1/λ) · ln(U)
        """
        # λ debe ser positivo para que la distribución tenga sentido
        if lam <= 0:
            raise ValueError(f"λ debe ser positivo. Recibido: λ={lam}.")
        # Transformada inversa de la exponencial
        return -(1.0 / lam) * math.log(self._lcg.next_uniform())

    def bernoulli(self, p: float) -> bool:
        """True con probabilidad p."""
        # Retorna True si el uniforme cae por debajo de p
        return self._lcg.next_uniform() < p

    def discrete_choice(self, values: list, probs: list):
        """
        Elige un valor de `values` según `probs` (deben sumar 1).
        Método de acumulación de probabilidades.
        """
        # Valida que las probabilidades formen una distribución válida
        if abs(sum(probs) - 1.0) > 1e-9:
            raise ValueError("Las probabilidades deben sumar 1.")

        # Genera un valor uniforme para seleccionar el intervalo correspondiente
        u, cumulative = self._lcg.next_uniform(), 0.0

        # Recorre los valores acumulando probabilidades
        for val, p in zip(values, probs):
            cumulative += p
            if u < cumulative:
                return val

        # Fallback por posibles errores de redondeo: devuelve el último valor
        return values[-1]