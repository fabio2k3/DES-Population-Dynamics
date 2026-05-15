"""
Módulo: Visualización

Genera todas las gráficas del proyecto usando matplotlib.

Gráficas disponibles:
  1. plot_population_evolution  — evolución total de la población
  2. plot_fm_ratio              — ratio F/M a lo largo del tiempo
  3. plot_births_deaths_decade  — nacimientos vs muertes por década
  4. plot_age_distribution      — histogramas de edad en t=0, t=50, t=100
  5. plot_multi_run_band        — trayectoria media ± 1σ (multi-corrida)
  6. plot_all                   — genera y guarda todas las gráficas

Uso:
    from src.visualizacion import plot_all
    plot_all(summary, multi=multi_analysis, output_dir="resultados/")
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING

# Configura matplotlib para generar archivos sin abrir ventana gráfica
import matplotlib
matplotlib.use("Agg")   # sin ventana gráfica, solo ficheros
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Importaciones solo para tipado estático
if TYPE_CHECKING:
    from src.estadisticas import RunSummary, MultiRunAnalysis

# Paleta consistente en todo el proyecto
# Cada color identifica un tipo de dato o serie de la visualización
C_TOTAL  = "#2E86AB"   # azul     — población total
C_FEMALE = "#E84855"   # rojo     — mujeres
C_MALE   = "#3BB273"   # verde    — hombres
C_BIRTH  = "#F4A261"   # naranja  — nacimientos
C_DEATH  = "#6D6875"   # púrpura  — muertes
C_BAND   = "#AED9E0"   # celeste  — banda de varianza


def _save(fig: plt.Figure, path: str) -> None:
    # Asegura que el directorio destino exista antes de guardar
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Guarda la figura con buena resolución y márgenes ajustados
    fig.savefig(path, dpi=150, bbox_inches="tight")
    # Cierra la figura para liberar memoria
    plt.close(fig)
    print(f"  Guardada: {path}")


# ==============================
# 1. EVOLUCIÓN DE LA POBLACIÓN =
# ==============================

def plot_population_evolution(summary: "RunSummary", path: str) -> None:
    # Extrae años y conteos desde los snapshots
    years   = [s.year        for s in summary.snapshots]
    total   = [s.total_alive for s in summary.snapshots]
    females = [s.alive_f     for s in summary.snapshots]
    males   = [s.alive_m     for s in summary.snapshots]

    # Crea la figura principal
    fig, ax = plt.subplots(figsize=(9, 5))

    # Línea total de población viva
    ax.plot(years, total,   color=C_TOTAL,  linewidth=2.5, marker="o",
            markersize=5, label="Total vivos")

    # Línea de mujeres vivas
    ax.plot(years, females, color=C_FEMALE, linewidth=1.8, marker="s",
            markersize=4, linestyle="--", label="Mujeres")

    # Línea de hombres vivos
    ax.plot(years, males,   color=C_MALE,   linewidth=1.8, marker="^",
            markersize=4, linestyle="--", label="Hombres")

    # Etiquetas y formato del gráfico
    ax.set_xlabel("Año de simulación", fontsize=11)
    ax.set_ylabel("Personas vivas",    fontsize=11)
    ax.set_title("Evolución de la Población", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Fuerza mostrar un tick por snapshot
    ax.set_xticks(years)

    fig.tight_layout()
    _save(fig, path)


# ======================================
# 2. RATIO F/M A LO LARGO DEL TIEMPO   =
# ======================================

def plot_fm_ratio(summary: "RunSummary", path: str) -> None:
    # Extrae años y ratio femenino/total de cada snapshot
    years  = [s.year     for s in summary.snapshots]
    ratios = [s.ratio_fm for s in summary.snapshots]

    fig, ax = plt.subplots(figsize=(9, 4))

    # Traza la evolución del ratio F/Total
    ax.plot(years, ratios, color=C_FEMALE, linewidth=2.5, marker="o", markersize=5)

    # Línea de equilibrio 0.5
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2,
               label="Equilibrio (0.5)")

    # Sombrea la zona donde hay mayoría femenina
    ax.fill_between(years, ratios, 0.5,
                    where=[r > 0.5 for r in ratios],
                    alpha=0.15, color=C_FEMALE, label="Mayoría femenina")

    # Sombrea la zona donde hay mayoría masculina
    ax.fill_between(years, ratios, 0.5,
                    where=[r <= 0.5 for r in ratios],
                    alpha=0.15, color=C_MALE, label="Mayoría masculina")

    ax.set_xlabel("Año de simulación", fontsize=11)
    ax.set_ylabel("Ratio F / Total",   fontsize=11)
    ax.set_title("Ratio Femenino/Total a lo Largo del Tiempo",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 0.7)
    ax.set_xticks(years)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, path)


# ======================================
# 3. NACIMIENTOS Y MUERTES POR DÉCADA  =
# ======================================

def plot_births_deaths_decade(summary: "RunSummary", path: str) -> None:
    # Etiquetas de cada década del horizonte de simulación
    decades = [f"{i*10}–{i*10+10}" for i in range(10)]
    births  = summary.births_per_decade
    deaths  = summary.deaths_per_decade

    # Posiciones de las barras
    x     = np.arange(len(decades))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))

    # Barras de nacimientos
    bars_b = ax.bar(x - width/2, births, width, color=C_BIRTH,
                    label="Nacimientos", edgecolor="white", linewidth=0.5)

    # Barras de muertes
    bars_d = ax.bar(x + width/2, deaths, width, color=C_DEATH,
                    label="Muertes",     edgecolor="white", linewidth=0.5)

    # Etiqueta numérica encima de cada barra no nula
    for bar in list(bars_b) + list(bars_d):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    str(int(h)), ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Década",              fontsize=11)
    ax.set_ylabel("Número de eventos",   fontsize=11)
    ax.set_title("Nacimientos y Muertes por Década",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(decades, rotation=30, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, path)


# =================================================
# 4. DISTRIBUCIÓN DE EDADES EN t=0, t=50, t=100   =
# =================================================

def plot_age_distribution(summary: "RunSummary", path: str) -> None:
    # Conjunto de datos a comparar, junto con su etiqueta y color asociado
    datasets = [
        (summary.age_dist_t0,   "t = 0 años",   C_TOTAL,  0.55),
        (summary.age_dist_t50,  "t = 50 años",  C_FEMALE, 0.55),
        (summary.age_dist_t100, "t = 100 años", C_MALE,   0.55),
    ]

    # Tres histogramas en una sola figura
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    bins = list(range(0, 126, 10))

    for ax, (data, label, color, alpha) in zip(axes, datasets):
        if data:
            # Histograma de edades
            ax.hist(data, bins=bins, color=color, alpha=alpha,
                    edgecolor="white", linewidth=0.5)

            # Marca la media de la distribución con una línea vertical
            ax.axvline(np.mean(data), color="black", linestyle="--",
                       linewidth=1.2, label=f"Media: {np.mean(data):.1f}")
            ax.legend(fontsize=9)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Edad (años)", fontsize=10)
        ax.set_ylabel("Frecuencia",  fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Distribución de Edades de la Población",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


# ======================================================
# 5. TRAYECTORIA MULTI-CORRIDA CON BANDA DE VARIANZA   =
# ======================================================

def plot_multi_run_band(multi: "MultiRunAnalysis", path: str) -> None:
    # Obtiene años, medias y desviaciones estándar desde el análisis multi-corrida
    years, means, stds = multi.pop_trajectory()
    means = np.array(means)
    stds  = np.array(stds)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Línea de la media poblacional entre corridas
    ax.plot(years, means, color=C_TOTAL, linewidth=2.5,
            marker="o", markersize=5, label=f"Media ({multi.n_runs} corridas)")

    # Banda de una desviación estándar
    ax.fill_between(years, means - stds, means + stds,
                    color=C_BAND, alpha=0.5, label="±1σ")

    # Banda de dos desviaciones estándar
    ax.fill_between(years, means - 2*stds, means + 2*stds,
                    color=C_BAND, alpha=0.2, label="±2σ")

    ax.set_xlabel("Año de simulación", fontsize=11)
    ax.set_ylabel("Personas vivas",    fontsize=11)
    ax.set_title(f"Evolución Poblacional — {multi.n_runs} Corridas Independientes",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)

    fig.tight_layout()
    _save(fig, path)


# =================================
# 6. GENERAR TODAS LAS GRÁFICAS   =
# =================================

def plot_all(summary: "RunSummary",
             multi: "MultiRunAnalysis | None" = None,
             output_dir: str = "resultados/") -> None:
    # Mensaje informativo de inicio
    print(f"\n  Generando gráficas en '{output_dir}'...")

    # Genera la gráfica de evolución poblacional
    plot_population_evolution(
        summary, os.path.join(output_dir, "01_evolucion_poblacion.png"))

    # Genera la gráfica del ratio femenino/total
    plot_fm_ratio(
        summary, os.path.join(output_dir, "02_ratio_fm.png"))

    # Genera nacimientos vs muertes por década
    plot_births_deaths_decade(
        summary, os.path.join(output_dir, "03_nacimientos_muertes_decada.png"))

    # Genera histogramas de edades en los instantes pedidos
    plot_age_distribution(
        summary, os.path.join(output_dir, "04_distribucion_edades.png"))

    # Si existe análisis multi-corrida, también genera la banda de varianza
    if multi is not None:
        plot_multi_run_band(
            multi, os.path.join(output_dir, "05_multi_corrida.png"))

    print("  ✔ Todas las gráficas generadas.\n")