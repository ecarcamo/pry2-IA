import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import time


# ─── Paleta de colores ────────────────────────────────────────────────────────
#  0 = pasillo libre     1 = pared         2 = inicio
#  3 = meta              4 = camino        5 = explorado (frío)
#  6 = frontera activa   (animación)
# Paleta de colores Modern Dashboard (Midnight Slate)
MAZE_COLORS = [
    "#0F172A",  # 0 pasillo libre (slate 900)
    "#334155",  # 1 pared (slate 700)
    "#10B981",  # 2 inicio (emerald)
    "#F43F5E",  # 3 meta (rose)
    "#06B6D4",  # 4 camino óptimo (cyan)
    "#1E40AF",  # 5 explorado (blue-800)
    "#F59E0B",  # 6 frontera activa / animando (amber)
]

MAZE_CMAP = ListedColormap(MAZE_COLORS)
MAZE_NORM = BoundaryNorm(boundaries=list(range(len(MAZE_COLORS) + 1)), ncolors=len(MAZE_COLORS))

# Colormap para el mapa de calor: blanco → amarillo → naranja → rojo oscuro
HEAT_CMAP = plt.cm.get_cmap("YlOrRd")


# ─── Mapa de calor ────────────────────────────────────────────────────────────

def heatmap_maze(maze, explored_order, path=None, algo_name=""):
    """
    Mapa de calor del orden de exploración.
    Amarillo claro = explorado primero · Rojo oscuro = explorado al final.
    Paredes en azul oscuro. Camino solución en azul translúcido.
    """
    rows, cols = maze.rows, maze.cols
    grid = np.array(maze.grid)
    total = len(explored_order)

    # ── Matriz de intensidad (NaN = no explorado ni pared) ───────────────────
    heat = np.full((rows, cols), np.nan)
    for i, (r, c) in enumerate(explored_order):
        heat[r][c] = (i + 1) / max(total, 1)   # 0 < valor ≤ 1.0

    # ── Figura ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0F0F23")
    ax.set_facecolor("#0F0F23")

    # Capa 1: pasillo libre sin explorar (gris muy oscuro)
    free_bg = np.where((grid == 0) & np.isnan(heat), 0.02, np.nan)
    ax.imshow(
        np.ma.masked_invalid(free_bg),
        cmap="Greys", vmin=0, vmax=1,
        interpolation="nearest", alpha=0.4,
    )

    # Capa 2: paredes (color sólido #334155)
    wall_mask = np.ma.masked_where(grid != 1, np.ones((rows, cols)))
    wall_cmap = ListedColormap(["#334155"])
    ax.imshow(wall_mask, cmap=wall_cmap, vmin=0.5, vmax=1.5,
              interpolation="nearest")

    # Capa 3: mapa de calor de exploración
    heat_img = ax.imshow(
        np.ma.masked_invalid(heat),
        cmap=HEAT_CMAP, vmin=0, vmax=1,
        interpolation="nearest", alpha=0.92,
    )

    # Capa 4: camino solución (cian translucido)
    if path:
        path_mat = np.full((rows, cols), np.nan)
        for r, c in path:
            path_mat[r][c] = 1.0
        path_cmap = ListedColormap(["#06B6D4"])
        ax.imshow(np.ma.masked_invalid(path_mat), cmap=path_cmap,
                  vmin=0.5, vmax=1.5, interpolation="nearest", alpha=0.75)

    # Marcadores de inicio y meta
    sr, sc = maze.start
    gr, gc = maze.goal
    ax.plot(sc, sr, "o", color="#10B981", markersize=14,
            markeredgecolor="white", markeredgewidth=1.5, zorder=10)
    ax.plot(gc, gr, "*", color="#F43F5E", markersize=16,
            markeredgecolor="white", markeredgewidth=1.5, zorder=10)

    # Colorbar
    cbar = fig.colorbar(heat_img, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["Primero", "25%", "50%", "75%", "Último"])
    cbar.ax.yaxis.set_tick_params(color="#F8FAFC", labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#F8FAFC")
    cbar.set_label("Orden de exploración", color="#F8FAFC", fontsize=9)
    cbar.outline.set_edgecolor("#334155")

    # Leyenda manual
    legend_elements = [
        mpatches.Patch(facecolor="#FFFF80", label="Explorado primero"),
        mpatches.Patch(facecolor="#CC0000", label="Explorado último"),
        mpatches.Patch(facecolor="#06B6D4", label="Camino óptimo"),
        mpatches.Patch(facecolor="#334155", label="Pared"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
              framealpha=0.85, facecolor="#1E293B",
              labelcolor="#94A3B8", edgecolor="#334155")

    title = f"Mapa de Calor — Orden de Exploración"
    if algo_name:
        title += f"  [{algo_name}]"
    ax.set_title(title, color="#F8FAFC", fontsize=11, pad=8, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0.4)
    return fig


# ─── Internos ─────────────────────────────────────────────────────────────────

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

    sr, sc = maze.start
    gr, gc = maze.goal
    display[sr][sc] = 2
    display[gr][gc] = 3

    return display


def _add_legend(ax, show_explored=False, show_frontier=False):
    elements = [
        mpatches.Patch(facecolor="#10B981", edgecolor="#10B981", label="Inicio"),
        mpatches.Patch(facecolor="#F43F5E", edgecolor="#F43F5E", label="Meta"),
        mpatches.Patch(facecolor="#06B6D4", edgecolor="#06B6D4", label="Camino"),
        mpatches.Patch(facecolor="#334155", edgecolor="#334155", label="Pared"),
    ]
    if show_explored:
        elements.insert(2, mpatches.Patch(facecolor="#1E40AF", edgecolor="#1E40AF", label="Explorado"))
    if show_frontier:
        elements.insert(2, mpatches.Patch(facecolor="#F59E0B", edgecolor="#F59E0B", label="Frontera"))
    ax.legend(
        handles=elements,
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
        facecolor="#1E293B",
        labelcolor="#94A3B8",
        edgecolor="#334155",
    )

def _render_frame(display, title="", title_color="#F8FAFC", figsize=6,
                  show_explored=False, show_frontier=False):
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.imshow(display, cmap=MAZE_CMAP, vmin=0, vmax=6, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=title_color, fontsize=10, pad=4, fontweight="bold")
    _add_legend(ax, show_explored=show_explored, show_frontier=show_frontier)
    plt.tight_layout(pad=0.3)
    return fig


# ─── API pública ──────────────────────────────────────────────────────────────

def visualize_maze(maze, path=None, explored=None, show_explored=True, figsize=7):
    display = _build_display(maze, path, explored if show_explored else None)
    fig = _render_frame(
        display, figsize=figsize,
        show_explored=show_explored and bool(explored),
    )
    return fig


def animate_solution(maze, path, explored, container, speed=30):
    """
    Animación en tres fases:
      1. Exploración: nodos recién descubiertos aparecen en ROJO (frontera activa)
         y los anteriores "se enfrían" a AMARILLO — muestra el frente de búsqueda.
      2. Transición: la frontera se apaga, todo queda en amarillo.
      3. Camino: el path óptimo se traza en AZUL sobre los explorados.
    """
    grid = np.array(maze.grid)
    display = grid.copy().astype(float)

    sr, sc = maze.start
    gr, gc = maze.goal
    display[sr][sc] = 2
    display[gr][gc] = 3

    delay = max(0.004, 1.0 / (speed * 6))

    explored_list = list(explored) if explored else []
    total = len(explored_list)

    # Número de frames de exploración (máx 180 para laberintos grandes)
    max_frames  = 180
    step_size   = max(1, total // max_frames)

    # Ventana de "frontera roja": cuántos nodos recientes se pintan de rojo
    frontier_win = max(step_size, total // 25)

    # ── Fase 1: exploración con frontera roja ────────────────────────────────
    for i in range(0, total, step_size):
        batch = explored_list[i: i + step_size]

        # Marcar batch actual como explorado (amarillo en display permanente)
        for r, c in batch:
            if display[r][c] == 0:
                display[r][c] = 5

        # Copia del frame con la frontera activa en rojo
        frame = display.copy()
        frontier_start = max(0, i - frontier_win + step_size)
        for r, c in explored_list[frontier_start: i + step_size]:
            if frame[r][c] == 5:
                frame[r][c] = 6      # rojo activo

        # Restaurar inicio/meta (no pintar encima)
        frame[sr][sc] = 2
        frame[gr][gc] = 3

        progress = min(100, int((i + step_size) / total * 100))
        fig = _render_frame(
            frame,
            title=f"Explorando...  {progress}%  ({min(i + step_size, total)}/{total} nodos)",
            title_color="#F59E0B",
            show_explored=True,
            show_frontier=True,
        )
        container.pyplot(fig)
        plt.close(fig)
        time.sleep(delay)

    # ── Fase 1b: apagar la frontera (breve pausa con todo en amarillo) ────────
    frame_final = display.copy()
    frame_final[sr][sc] = 2
    frame_final[gr][gc] = 3
    fig = _render_frame(
        frame_final,
        title=f"Exploración completa — {total} nodos visitados",
        title_color="#60A5FA",
        show_explored=True,
    )
    container.pyplot(fig)
    plt.close(fig)
    time.sleep(delay * 8)

    # ── Fase 2: trazar el camino en azul ─────────────────────────────────────
    path_display = frame_final.copy()
    for step_idx, step in enumerate(path):
        r, c = step
        path_display[r][c] = 4
        path_display[sr][sc] = 2
        path_display[gr][gc] = 3

        fig = _render_frame(
            path_display,
            title=f"Trazando camino óptimo... ({step_idx + 1}/{len(path)})",
            title_color="#06B6D4",
            show_explored=True,
        )
        container.pyplot(fig)
        plt.close(fig)
        time.sleep(delay * 3)

    # Frame final con leyenda completa
    fig = _render_frame(
        path_display,
        title=f"Camino encontrado — {len(path)} pasos",
        title_color="#10B981",
        show_explored=True,
    )
    container.pyplot(fig)
    plt.close(fig)


# ─── Visualización interactiva con Plotly ─────────────────────────────────────

def visualize_maze_plotly(maze, path=None, explored=None, show_explored=True):
    """
    Renderiza el laberinto como un heatmap interactivo de Plotly.
    Hover muestra: (fila, col), tipo de celda, orden de exploración y paso en el path.
    """
    import plotly.graph_objects as go

    rows, cols = maze.rows, maze.cols
    grid = np.array(maze.grid)

    # ── Índice de exploración: {(r,c): orden} ────────────────────────────────
    explored_list = list(explored) if explored else []
    explore_idx   = {(r, c): i + 1 for i, (r, c) in enumerate(explored_list)}

    # ── Índice del camino: {(r,c): paso} ─────────────────────────────────────
    path_list = list(path) if path else []
    path_idx  = {(r, c): i + 1 for i, (r, c) in enumerate(path_list)}

    # ── Construir matriz de valores y textos de hover ────────────────────────
    z     = grid.copy().astype(float)
    hover = np.empty((rows, cols), dtype=object)

    sr, sc = maze.start
    gr, gc = maze.goal

    # Distancia Manhattan precalculada al goal
    def _mdist(r, c):
        return abs(r - gr) + abs(c - gc)

    CELL_LABELS = {
        0: "Pasillo libre",
        1: "Pared",
        2: "Inicio",
        3: "Meta",
        4: "Camino",
        5: "Explorado",
    }

    for r in range(rows):
        for c in range(cols):
            v    = int(grid[r][c])
            tip  = CELL_LABELS.get(v, "?")
            dist = _mdist(r, c)
            hover[r][c] = (
                f"<b>({r}, {c})</b><br>"
                f"Tipo: {tip}<br>"
                f"Dist. Manhattan a meta: {dist}"
            )

    # Marcar explorados
    if show_explored and explored_list:
        for r, c in explored_list:
            if z[r][c] == 0:
                z[r][c] = 5
                hover[r][c] = (
                    f"<b>({r}, {c})</b><br>"
                    f"Tipo: Explorado<br>"
                    f"Orden de exploración: #{explore_idx[(r,c)]}<br>"
                    f"Dist. Manhattan a meta: {_mdist(r,c)}"
                )

    # Marcar camino
    for r, c in path_list:
        if grid[r][c] == 0:
            z[r][c] = 4
            hover[r][c] = (
                f"<b>({r}, {c})</b><br>"
                f"Tipo: Camino<br>"
                f"Paso: {path_idx[(r,c)]} / {len(path_list)}<br>"
                f"Dist. Manhattan a meta: {_mdist(r,c)}"
            )

    # Inicio / meta
    z[sr][sc] = 2
    z[gr][gc] = 3
    hover[sr][sc] = f"<b>({sr}, {sc})</b><br>Tipo: Inicio"
    hover[gr][gc] = f"<b>({gr}, {gc})</b><br>Tipo: Meta"

    # ── Colorscale discreta (6 bandas, una por valor 0–5) ────────────────────
    # Normalizado al rango [0, 1] sobre vmax=5
    _cs = [
        [0.000, "#0F172A"], [0.199, "#0F172A"],   # 0 pasillo
        [0.200, "#334155"], [0.399, "#334155"],   # 1 pared
        [0.400, "#10B981"], [0.599, "#10B981"],   # 2 inicio
        [0.600, "#F43F5E"], [0.799, "#F43F5E"],   # 3 meta
        [0.800, "#06B6D4"], [0.899, "#06B6D4"],   # 4 camino
        [0.900, "#1E40AF"], [1.000, "#1E40AF"],   # 5 explorado
    ]

    fig = go.Figure()

    # Capa 1: heatmap del laberinto
    fig.add_trace(go.Heatmap(
        z=z,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale=_cs,
        zmin=0, zmax=5,
        showscale=False,
        xgap=0.3,
        ygap=0.3,
    ))

    # Capa 2: marcadores de inicio y meta (más visibles que el heatmap)
    fig.add_trace(go.Scatter(
        x=[sc, gc],
        y=[sr, gr],
        mode="markers+text",
        marker=dict(
            symbol=["circle", "star"],
            size=[16, 18],
            color=["#10B981", "#F43F5E"],
            line=dict(color="white", width=2),
        ),
        text=["S", "G"],
        textfont=dict(color="white", size=10, family="Arial Black"),
        textposition="middle center",
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#94A3B8", family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=620,
        title=dict(
            text="Laberinto interactivo — pasa el cursor sobre las celdas",
            font=dict(color="#F8FAFC", size=14),
            x=0.5,
        ),
        xaxis=dict(
            showticklabels=False, showgrid=False,
            zeroline=False, range=[-0.5, cols - 0.5],
        ),
        yaxis=dict(
            showticklabels=False, showgrid=False,
            zeroline=False, autorange="reversed",
            range=[-0.5, rows - 0.5],
            scaleanchor="x", scaleratio=1,
        ),
    )
    return fig