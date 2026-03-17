import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import time


# Paleta de colores semántica para el laberinto
MAZE_COLORS = [
    "#F5F5F0",  # 0 = pasillo libre
    "#1A1A2E",  # 1 = pared
    "#00C853",  # 2 = inicio
    "#FF6D00",  # 3 = meta
    "#00B0FF",  # 4 = camino solución
    "#FFD54F",  # 5 = nodos explorados
]

MAZE_CMAP = ListedColormap(MAZE_COLORS)
MAZE_NORM = BoundaryNorm(boundaries=[0, 1, 2, 3, 4, 5, 6], ncolors=6)


def _build_display(maze, path=None, explored=None):
    grid = np.array(maze.grid)
    display = grid.copy().astype(float)

    if explored:
        for r, c in explored:
            if display[r][c] == 0:
                display[r][c] = 5

    if path:
        for r, c in path:
            display[r][c] = 4

    start_r, start_c = maze.start
    goal_r, goal_c = maze.goal

    display[start_r][start_c] = 2
    display[goal_r][goal_c] = 3

    return display


def _add_legend(ax, show_explored=False):
    legend_elements = [
        mpatches.Patch(facecolor="#00C853", edgecolor="#00C853", label="Inicio"),
        mpatches.Patch(facecolor="#FF6D00", edgecolor="#FF6D00", label="Meta"),
        mpatches.Patch(facecolor="#00B0FF", edgecolor="#00B0FF", label="Camino"),
        mpatches.Patch(facecolor="#1A1A2E", edgecolor="#555", label="Pared"),
    ]
    if show_explored:
        legend_elements.insert(2, mpatches.Patch(
            facecolor="#FFD54F", edgecolor="#FFD54F", label="Explorado"
        ))
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
        facecolor="#1A1A2E",
        labelcolor="#E8E8F0",
        edgecolor="#444",
    )


def visualize_maze(maze, path=None, explored=None, show_explored=True, figsize=7):
    display = _build_display(maze, path, explored if show_explored else None)

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    fig.patch.set_facecolor("#0F0F23")
    ax.set_facecolor("#0F0F23")

    ax.imshow(display, cmap=MAZE_CMAP, vmin=0, vmax=5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    _add_legend(ax, show_explored=show_explored and bool(explored))

    plt.tight_layout(pad=0.3)
    return fig


def animate_solution(maze, path, explored, container, speed=30):
    """
    Animación en dos fases:
      1. Exploración: pinta nodos visitados en amarillo progresivamente
      2. Solución: pinta el path óptimo en azul
    """
    grid = np.array(maze.grid)
    display = grid.copy().astype(float)

    start_r, start_c = maze.start
    goal_r, goal_c = maze.goal

    display[start_r][start_c] = 2
    display[goal_r][goal_c] = 3

    delay = max(0.005, 1.0 / (speed * 5))

    # — Fase 1: exploración —
    explored_list = list(explored) if explored else []
    step_size = max(1, len(explored_list) // 150)  # limitar frames

    for i in range(0, len(explored_list), step_size):
        batch = explored_list[i: i + step_size]
        for r, c in batch:
            if display[r][c] == 0:
                display[r][c] = 5

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#0F0F23")
        ax.set_facecolor("#0F0F23")
        ax.imshow(display, cmap=MAZE_CMAP, vmin=0, vmax=5, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Explorando...", color="#FFD54F", fontsize=10, pad=4)
        container.pyplot(fig)
        plt.close(fig)
        time.sleep(delay)

    # — Fase 2: camino solución —
    for step in path:
        r, c = step
        display[r][c] = 4
        display[start_r][start_c] = 2
        display[goal_r][goal_c] = 3

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#0F0F23")
        ax.set_facecolor("#0F0F23")
        ax.imshow(display, cmap=MAZE_CMAP, vmin=0, vmax=5, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Trazando camino...", color="#00B0FF", fontsize=10, pad=4)
        container.pyplot(fig)
        plt.close(fig)
        time.sleep(delay * 2)