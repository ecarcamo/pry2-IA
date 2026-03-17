import streamlit as st
import pandas as pd

from maze.loader import find_start_goal
from maze.maze import Maze

from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.greedy import greedy
from algorithms.astar import astar

from heuristics.manhattan import manhattan
from heuristics.euclidean import euclidean

from utils.metrics import measure_algorithm
from utils.visualization import visualize_maze, animate_solution
from utils.charts import (
    path_bar_chart,
    explored_bar_chart,
    runtime_bar_chart,
    multi_bar_chart,
)


# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Maze Solver · IA",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS adicional para pulir detalles visuales
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        .stMetric { background-color: #1A1A3E; border-radius: 10px; padding: 0.6rem 1rem; }
        .stMetric label { color: #A0A8D0 !important; font-size: 0.78rem; }
        .stMetric [data-testid="metric-container"] > div:nth-child(2) { font-size: 1.5rem; color: #E8E8F0; font-weight: 700; }
        .algo-badge { display:inline-block; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; }
        div[data-testid="stSidebar"] { border-right: 1px solid #2A2A5A; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧩 Maze Solver")
    st.markdown("*Visualizador de algoritmos de búsqueda*")
    st.divider()

    st.markdown("### 📂 Laberinto")
    uploaded_file = st.file_uploader(
        "Cargar archivo `.txt`",
        type=["txt"],
        help="Formato: matriz de 0s (paso libre), 1s (pared), 2 (inicio), 3 (meta)",
    )

    st.markdown("### ⚙️ Algoritmo")
    algorithm_name = st.selectbox(
        "Algoritmo",
        ["BFS", "DFS", "Greedy", "A*"],
        help="BFS y DFS no usan heurística. Greedy y A* sí.",
    )

    heuristic_name = st.selectbox(
        "Heurística",
        ["Manhattan", "Euclidean"],
        help="Aplica solo a Greedy y A*",
        disabled=algorithm_name in ["BFS", "DFS"],
    )

    st.markdown("### 🎨 Visualización")
    show_explored = st.checkbox("Mostrar nodos explorados", value=True)
    animate = st.checkbox("Animar solución")
    if animate:
        speed = st.slider("Velocidad de animación", min_value=1, max_value=100, value=30)
    else:
        speed = 30

    st.divider()
    run_button = st.button("▶ Ejecutar Solver", width="stretch", type="primary")

    st.markdown("---")
    st.caption("Proyecto 2 · Inteligencia Artificial")


# ─── Contenido principal ─────────────────────────────────────────────────────
st.markdown("# 🧩 Maze Solver")
st.markdown("Visualización del proceso de búsqueda de algoritmos en laberintos")

if not uploaded_file:
    st.info("👈 Sube un archivo de laberinto en el panel lateral para comenzar.", icon="📂")
    st.stop()

# ── Parsear laberinto ─────────────────────────────────────────────────────────
maze_grid = []
for line in uploaded_file:
    row = [int(c) for c in line.decode().strip() if c in "01234"]
    if row:
        maze_grid.append(row)

start, goal = find_start_goal(maze_grid)

if start is None or goal is None:
    st.error("❌ El laberinto debe contener un punto de inicio (2) y una meta (3).")
    st.stop()

maze = Maze(maze_grid, start, goal)

if heuristic_name == "Manhattan":
    heuristic = manhattan
else:
    heuristic = euclidean

# ── Ejecutar algoritmo seleccionado ──────────────────────────────────────────
if run_button:
    st.session_state["ran"] = True

    with st.spinner(f"Ejecutando {algorithm_name}..."):
        algo_map = {
            "BFS":    lambda m: bfs(m),
            "DFS":    lambda m: dfs(m),
            "Greedy": lambda m: greedy(m, heuristic),
            "A*":     lambda m: astar(m, heuristic),
        }
        result = measure_algorithm(algo_map[algorithm_name], maze)

        # Comparación entre todos los algoritmos
        all_results = []
        for name, algo in algo_map.items():
            r = measure_algorithm(algo, maze)
            all_results.append({
                "Algorithm":       name,
                "Path Length":     r["path_length"],
                "Nodes Explored":  r["nodes_explored"],
                "Runtime (s)":     round(r["runtime"], 6),
                "explored":        r["explored"],
                "path":            r["path"],
            })

    st.session_state["result"]      = result
    st.session_state["all_results"] = all_results
    st.session_state["maze"]        = maze
    st.session_state["algo_name"]   = algorithm_name
    st.session_state["heuristic"]   = heuristic_name

# ── Mostrar resultados ────────────────────────────────────────────────────────
if st.session_state.get("ran") and "result" in st.session_state:
    result      = st.session_state["result"]
    all_results = st.session_state["all_results"]
    maze        = st.session_state["maze"]

    total_cells    = maze.rows * maze.cols
    explored_ratio = result["nodes_explored"] / total_cells * 100
    efficiency     = result["path_length"] / max(result["nodes_explored"], 1) * 100

    tab1, tab2, tab3 = st.tabs(["🗺 Solución", "📊 Métricas", "⚖️ Comparación"])

    # ══════════════════════════════════════════════
    # TAB 1 — Solución visual
    # ══════════════════════════════════════════════
    with tab1:
        col_info, col_maze = st.columns([1, 2.5])

        with col_info:
            st.markdown("#### ℹ️ Info del laberinto")
            st.markdown(f"**Tamaño:** {maze.rows} × {maze.cols}")
            st.markdown(f"**Celdas totales:** `{total_cells}`")
            st.markdown(f"**Inicio:** `{maze.start}`")
            st.markdown(f"**Meta:** `{maze.goal}`")

            algo_used = st.session_state.get("algo_name", algorithm_name)
            heur_used = st.session_state.get("heuristic", heuristic_name)
            st.markdown(f"**Algoritmo:** `{algo_used}`")
            if algo_used in ["Greedy", "A*"]:
                st.markdown(f"**Heurística:** `{heur_used}`")

            st.divider()
            st.markdown("#### 📌 Resultados")
            st.metric("📏 Path Length",      result["path_length"])
            st.metric("🔍 Nodos Explorados", result["nodes_explored"])
            st.metric("⏱ Runtime",          f"{result['runtime']:.5f} s")
            st.metric("📐 % Explorado",      f"{explored_ratio:.1f}%")
            st.metric("✅ Eficiencia",       f"{efficiency:.1f}%",
                      help="Path length / nodos explorados × 100")

        with col_maze:
            if animate:
                animation_container = st.empty()
                animate_solution(
                    maze,
                    result["path"],
                    result["explored"],
                    animation_container,
                    speed=speed,
                )
            else:
                fig = visualize_maze(
                    maze,
                    path=result["path"],
                    explored=result["explored"] if show_explored else None,
                    show_explored=show_explored,
                )
                st.pyplot(fig, width="stretch")

    # ══════════════════════════════════════════════
    # TAB 2 — Métricas del algoritmo seleccionado
    # ══════════════════════════════════════════════
    with tab2:
        st.markdown(f"### Métricas de **{st.session_state.get('algo_name', algorithm_name)}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📏 Path Length",      result["path_length"])
        m2.metric("🔍 Nodos Explorados", result["nodes_explored"])
        m3.metric("⏱ Runtime (s)",       f"{result['runtime']:.5f}")
        m4.metric("📐 % Explorado",      f"{explored_ratio:.1f}%")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Cobertura de exploración")
            explored_count = result["nodes_explored"]
            free_cells = sum(
                1 for row in maze.grid for v in row if v != 1
            )
            cov_data = pd.DataFrame({
                "Tipo":   ["Exploradas", "Sin explorar"],
                "Celdas": [explored_count, max(0, free_cells - explored_count)],
            })
            import altair as alt
            pie = (
                alt.Chart(cov_data)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("Celdas:Q"),
                    color=alt.Color(
                        "Tipo:N",
                        scale=alt.Scale(
                            domain=["Exploradas", "Sin explorar"],
                            range=["#FFD54F", "#2A2A4A"],
                        ),
                        legend=alt.Legend(labelColor="#E8E8F0", titleColor="#E8E8F0"),
                    ),
                    tooltip=["Tipo:N", "Celdas:Q"],
                )
                .properties(height=220, title=alt.TitleParams("Celdas libres exploradas", color="#E8E8F0"))
                .configure_view(strokeOpacity=0)
                .configure(background="#1A1A3E")
            )
            st.altair_chart(pie, width="stretch")

        with col_b:
            st.markdown("#### Relación Path vs Explorados")
            ratio_data = pd.DataFrame({
                "Tipo":   ["Path óptimo", "Nodos extra explorados"],
                "Nodos":  [result["path_length"],
                           max(0, result["nodes_explored"] - result["path_length"])],
            })
            bars = (
                alt.Chart(ratio_data)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X("Tipo:N", axis=alt.Axis(labelColor="#E8E8F0", title="")),
                    y=alt.Y("Nodos:Q", axis=alt.Axis(labelColor="#E8E8F0")),
                    color=alt.Color(
                        "Tipo:N",
                        scale=alt.Scale(
                            domain=["Path óptimo", "Nodos extra explorados"],
                            range=["#00B0FF", "#FF6D00"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Tipo:N", "Nodos:Q"],
                )
                .properties(height=220, title=alt.TitleParams("Path vs Overhead de exploración", color="#E8E8F0"))
                .configure_view(strokeOpacity=0)
                .configure(background="#1A1A3E", axis=alt.AxisConfig(gridColor="#2A2A4A"))
            )
            st.altair_chart(bars, width="stretch")

    # ══════════════════════════════════════════════
    # TAB 3 — Comparación entre algoritmos
    # ══════════════════════════════════════════════
    with tab3:
        st.markdown("### Comparación de los 4 algoritmos")
        st.caption(f"Heurística usada para Greedy y A*: **{st.session_state.get('heuristic', heuristic_name)}**")

        # DataFrame para tabla (sin columns internas como explored/path)
        df_display = pd.DataFrame([{
            "Algoritmo":        r["Algorithm"],
            "Path Length":      r["Path Length"],
            "Nodos Explorados": r["Nodes Explored"],
            "Runtime (s)":      r["Runtime (s)"],
        } for r in all_results])

        # Resaltado: verde para mínimo, rojo suave para máximo
        styled = df_display.style.highlight_min(
            subset=["Path Length", "Nodos Explorados", "Runtime (s)"],
            color="#1A4731",
        ).highlight_max(
            subset=["Path Length", "Nodos Explorados", "Runtime (s)"],
            color="#4A1A1A",
        ).format({"Runtime (s)": "{:.6f}"})

        st.dataframe(styled, width="stretch", hide_index=True)

        st.divider()

        df_chart = pd.DataFrame([{
            "Algorithm":      r["Algorithm"],
            "Path Length":    r["Path Length"],
            "Nodes Explored": r["Nodes Explored"],
            "Runtime (s)":    r["Runtime (s)"],
        } for r in all_results])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.altair_chart(path_bar_chart(df_chart), width="stretch")
        with c2:
            st.altair_chart(explored_bar_chart(df_chart), width="stretch")
        with c3:
            st.altair_chart(runtime_bar_chart(df_chart), width="stretch")

        st.divider()
        st.markdown("#### Vista comparativa de soluciones (grid 2×2)")

        grid_cols = st.columns(2)
        for idx, r in enumerate(all_results):
            with grid_cols[idx % 2]:
                algo_label = r["Algorithm"]
                badge_colors = {
                    "BFS":    "#00B0FF",
                    "DFS":    "#FF6D00",
                    "Greedy": "#00E676",
                    "A*":     "#FF4081",
                }
                color = badge_colors.get(algo_label, "#888")
                st.markdown(
                    f'<span class="algo-badge" style="background:{color}22;color:{color};border:1px solid {color}66">'
                    f'{algo_label}</span>',
                    unsafe_allow_html=True,
                )
                fig = visualize_maze(
                    maze,
                    path=r["path"],
                    explored=r["explored"] if show_explored else None,
                    show_explored=show_explored,
                    figsize=5,
                )
                st.pyplot(fig, width="stretch")
                path_len  = r["Path Length"]
                nodes_exp = r["Nodes Explored"]
                runtime_v = r["Runtime (s)"]
                st.caption(f"Path: **{path_len}** · Explorados: **{nodes_exp}** · Runtime: **{runtime_v:.5f}s**")