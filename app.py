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

from utils.metrics import measure_algorithm, branching_factor
from utils.visualization import visualize_maze, animate_solution, heatmap_maze, visualize_maze_plotly
from utils.charts import (
    path_bar_chart,
    explored_bar_chart,
    runtime_bar_chart,
    multi_bar_chart,
)


# ─── Cache de algoritmos ──────────────────────────────────────────────────────
# Los args deben ser hasheables: grid como tuple[tuple], heuristic como string.
@st.cache_data
def run_all_algorithms(
    grid: tuple,
    start: tuple,
    goal: tuple,
    heuristic_name: str,
) -> list:
    """Ejecuta los 4 algoritmos y retorna resultados. Se cachea por (grid, heuristic)."""
    heuristic = manhattan if heuristic_name == "Manhattan" else euclidean
    maze = Maze([list(row) for row in grid], start, goal)

    algo_map = {
        "BFS":    lambda m: bfs(m),
        "DFS":    lambda m: dfs(m),
        "Greedy": lambda m: greedy(m, heuristic),
        "A*":     lambda m: astar(m, heuristic),
    }

    results = []
    for name, algo in algo_map.items():
        r = measure_algorithm(algo, maze)
        results.append({
            "Algorithm":      name,
            "Path Length":    r["path_length"],
            "Nodes Explored": r["nodes_explored"],
            "Runtime (s)":    round(r["runtime"], 6),
            "explored":       r["explored"],
            "path":           r["path"],
        })
    return results


# ─── Benchmark con pares aleatorios ─────────────────────────────────────────────────────
@st.cache_data
def run_benchmark(
    grid: tuple,
    heuristic_name: str,
    n_experiments: int = 20,
    seed: int = 42,
) -> tuple:
    """Ejecuta n_experiments con pares (start, goal) aleatorios sobre el mismo grid.
    Retorna (df_stats, df_raw, valid_n) — cacheado por (grid, heuristic, n, seed).
    """
    import random, statistics as _st

    random.seed(seed)
    heuristic = manhattan if heuristic_name == "Manhattan" else euclidean
    grid_list = [list(row) for row in grid]

    # Celdas libres (valor 0) sobre las que se puede colocar start/goal
    free_cells = [
        (r, c)
        for r, row in enumerate(grid_list)
        for c, v in enumerate(row)
        if v == 0
    ]

    raw = {algo: {"runtime": [], "nodes": [], "path": []} for algo in ["BFS", "DFS", "Greedy", "A*"]}
    valid_n = 0
    attempts = 0

    while valid_n < n_experiments and attempts < n_experiments * 10:
        attempts += 1
        if len(free_cells) < 2:
            break
        exp_start, exp_goal = random.sample(free_cells, 2)
        exp_maze = Maze(grid_list, exp_start, exp_goal)

        algo_map = {
            "BFS":    lambda m: bfs(m),
            "DFS":    lambda m: dfs(m),
            "Greedy": lambda m: greedy(m, heuristic),
            "A*":     lambda m: astar(m, heuristic),
        }

        # Verificar conectividad con BFS antes de correr todo
        probe = measure_algorithm(lambda m: bfs(m), exp_maze)
        if not probe["path"]:
            continue

        valid_n += 1
        for name, algo in algo_map.items():
            r = measure_algorithm(algo, exp_maze)
            raw[name]["runtime"].append(r["runtime"])
            raw[name]["nodes"].append(r["nodes_explored"])
            raw[name]["path"].append(r["path_length"] if r["path"] else 0)

    # ── Estadísticas agregadas ──────────────────────────────────────────
    summary_rows = []
    raw_rows = []
    for algo, data in raw.items():
        n = len(data["runtime"])
        if n == 0:
            continue
        summary_rows.append({
            "Algoritmo":       algo,
            "Avg Runtime (s)": round(_st.mean(data["runtime"]), 6),
            "Std Runtime":     round(_st.stdev(data["runtime"]) if n > 1 else 0, 6),
            "Min Runtime":     round(min(data["runtime"]), 6),
            "Max Runtime":     round(max(data["runtime"]), 6),
            "Avg Nodos":       round(_st.mean(data["nodes"]), 1),
            "Std Nodos":       round(_st.stdev(data["nodes"]) if n > 1 else 0, 1),
            "Avg Path":        round(_st.mean(data["path"]), 1),
            "N":               n,
        })
        for i, (rt, nd, pl) in enumerate(zip(data["runtime"], data["nodes"], data["path"])):
            raw_rows.append({
                "Experimento": i + 1,
                "Algoritmo":   algo,
                "Runtime (s)": round(rt, 6),
                "Nodos":       nd,
                "Path Length": pl,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(raw_rows), valid_n


# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Maze Solver · IA",
    page_icon="🔹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS adicional para pulir detalles visuales (Estilo Dashboard Premium)
st.markdown(
    """
    <style>
        /* Ocultar menú y footer nativos de Streamlit, manteniendo el botón de la barra lateral */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Espaciado del lienzo principal */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Estilización moderna de las tarjetas de métricas (KPIs) */
        div[data-testid="metric-container"] {
            background: linear-gradient(145deg, #1e1e2f, #252538);
            border: 1px solid #2d2d44;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
        }
        
        /* Color y estilo del label de la métrica */
        div[data-testid="metric-container"] > div:nth-child(1) {
            color: #a0a0b8 !important;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Color del valor numérico de la métrica */
        div[data-testid="metric-container"] > div:nth-child(2) {
            color: #ffffff !important;
            font-weight: 700;
            font-size: 1.8rem;
        }

        /* Badges de algoritmos elegantes */
        .algo-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* Títulos y separadores más limpios */
        h1, h2, h3, h4 {
            color: #f8fafc;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        
        hr {
            border-color: #2d2d44;
        }
        
        /* Borde sutil del sidebar */
        div[data-testid="stSidebar"] { border-right: 1px solid #2d2d44; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Maze Solver")
    st.markdown("*Visualizador de algoritmos de búsqueda*")
    st.divider()

    st.markdown("### Laberinto")
    uploaded_file = st.file_uploader(
        "Cargar archivo `.txt`",
        type=["txt"],
        help="Formato: matriz de 0s (paso libre), 1s (pared), 2 (inicio), 3 (meta)",
    )

    st.markdown("### Algoritmo")
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

    st.markdown("### Visualización")
    show_explored  = st.checkbox("Mostrar nodos explorados", value=True)
    interactive    = st.checkbox("Vista interactiva (Plotly)", value=False,
                                help="Activa hover con info de cada celda: coordenadas, tipo, orden de exploración.")
    animate        = st.checkbox("Animar solución", disabled=interactive)
    if animate and not interactive:
        speed = st.slider("Velocidad de animación", min_value=1, max_value=100, value=30)
    else:
        speed = 30

    st.divider()
    run_button = st.button("Ejecutar Solver", width="stretch", type="primary")

    st.markdown("---")
    st.caption("Proyecto 2 · Inteligencia Artificial")


# ─── Contenido principal ─────────────────────────────────────────────────────
st.markdown("# Maze Solver")
st.markdown("Visualización del proceso de búsqueda de algoritmos en laberintos")

if not uploaded_file:
    st.info("Sube un archivo de laberinto en el panel lateral para comenzar.")
    st.stop()

# ── Parsear laberinto ─────────────────────────────────────────────────────────
maze_grid = []
for line in uploaded_file:
    row = [int(c) for c in line.decode().strip() if c in "01234"]
    if row:
        maze_grid.append(row)

start, goal = find_start_goal(maze_grid)

if start is None or goal is None:
    st.error("El laberinto debe contener un punto de inicio (2) y una meta (3).")
    st.stop()

maze = Maze(maze_grid, start, goal)

if heuristic_name == "Manhattan":
    heuristic = manhattan
else:
    heuristic = euclidean

# ── Ejecutar algoritmo seleccionado ──────────────────────────────────────────
if run_button:
    st.session_state["ran"]         = True
    st.session_state["algo_name"]   = algorithm_name
    st.session_state["heuristic"]   = heuristic_name
    st.session_state["maze"]        = maze

    # Convertir grid a tuple de tuples para que sea hasheable por cache_data
    grid_key = tuple(tuple(row) for row in maze_grid)

    with st.spinner("Ejecutando algoritmos..."):
        all_results = run_all_algorithms(grid_key, start, goal, heuristic_name)

    # Extraer el resultado del algoritmo seleccionado
    result = next(r for r in all_results if r["Algorithm"] == algorithm_name)
    # Reconstruir dict con las mismas claves que usaba measure_algorithm
    result_dict = {
        "path":           result["path"],
        "path_length":    result["Path Length"],
        "nodes_explored": result["Nodes Explored"],
        "explored":       result["explored"],
        "runtime":        result["Runtime (s)"],
    }

    st.session_state["result"]      = result_dict
    st.session_state["all_results"] = all_results

# ── Mostrar resultados ────────────────────────────────────────────────────────
if st.session_state.get("ran") and "result" in st.session_state:
    result      = st.session_state["result"]
    all_results = st.session_state["all_results"]
    maze        = st.session_state["maze"]

    total_cells    = maze.rows * maze.cols
    explored_ratio = result["nodes_explored"] / total_cells * 100
    efficiency     = result["path_length"] / max(result["nodes_explored"], 1) * 100
    bf             = branching_factor(result["nodes_explored"], result["path_length"])

    tab1, tab2, tab3, tab4 = st.tabs(["Solución", "Métricas", "Comparación", "Benchmark"])

    # ══════════════════════════════════════════════
    # TAB 1 — Solución visual
    # ══════════════════════════════════════════════
    with tab1:
        col_info, col_maze = st.columns([1, 2.5])

        with col_info:
            st.markdown("#### Información del laberinto")
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
            st.markdown("#### Resultados")
            st.metric("Longitud del camino", result["path_length"])
            st.metric("Nodos Explorados", result["nodes_explored"])
            st.metric("Tiempo de ejecución", f"{result['runtime']:.5f} s")
            st.metric("% Explorado", f"{explored_ratio:.1f}%")
            st.metric("Eficiencia", f"{efficiency:.1f}%",
                      help="Path length / nodos explorados × 100")
            st.metric("Factor de Ramificación", f"{bf:.4f}",
                      help="Factor de ramificación efectivo: N^(1/d)")

        with col_maze:
            if animate and not interactive:
                animation_container = st.empty()
                animate_solution(
                    maze,
                    result["path"],
                    result["explored"],
                    animation_container,
                    speed=speed,
                )
            elif interactive:
                try:
                    fig_plotly = visualize_maze_plotly(
                        maze,
                        path=result["path"],
                        explored=result["explored"] if show_explored else None,
                        show_explored=show_explored,
                    )
                    st.plotly_chart(fig_plotly, width="stretch")
                except ImportError:
                    st.error("Plotly no está instalado. Ejecuta: pip install plotly")
                    fig = visualize_maze(
                        maze, path=result["path"],
                        explored=result["explored"] if show_explored else None,
                        show_explored=show_explored,
                    )
                    st.pyplot(fig, width="stretch")
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

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Longitud del camino", result["path_length"])
        m2.metric("Nodos Explorados", result["nodes_explored"])
        m3.metric("Tiempo de ejecución", f"{result['runtime']:.5f}")
        m4.metric("% Explorado", f"{explored_ratio:.1f}%")
        m5.metric("Factor de Ramificación", f"{bf:.4f}",
                  help="Factor de ramificación efectivo b ≈ N^(1/d)")

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
                            range=["#1E40AF", "#475569"],
                        ),
                        legend=alt.Legend(labelColor="#F8FAFC", titleColor="#F8FAFC"),
                    ),
                    tooltip=["Tipo:N", "Celdas:Q"],
                )
                .properties(height=220, title=alt.TitleParams("Celdas libres exploradas", color="#F8FAFC"))
                .configure_view(strokeOpacity=0)
                .configure(background="transparent")
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
                    x=alt.X("Tipo:N", axis=alt.Axis(labelColor="#94A3B8", title="")),
                    y=alt.Y("Nodos:Q", axis=alt.Axis(labelColor="#94A3B8")),
                    color=alt.Color(
                        "Tipo:N",
                        scale=alt.Scale(
                            domain=["Path óptimo", "Nodos extra explorados"],
                            range=["#06B6D4", "#F43F5E"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Tipo:N", "Nodos:Q"],
                )
                .properties(height=220, title=alt.TitleParams("Path vs Overhead de exploración", color="#F8FAFC"))
                .configure_view(strokeOpacity=0)
                .configure(background="transparent", axis=alt.AxisConfig(gridColor="#334155", domainColor="#334155"))
            )
            st.altair_chart(bars, width="stretch")

        # ── Mapa de calor del algoritmo seleccionado ──────────────────────────
        st.divider()
        st.markdown("#### Mapa de Calor de Exploración")
        st.caption(
            "Muestra **en qué orden** el algoritmo descubrió cada celda. "
            "Amarillo = explorado primero · Rojo oscuro = explorado al final. "
            "Las celdas en gris nunca fueron alcanzadas."
        )

        st.markdown("**Algoritmo actual**")
        fig_heat = heatmap_maze(
            maze,
            result["explored"],
            path=result["path"],
            algo_name=st.session_state.get("algo_name", algorithm_name),
        )
        st.pyplot(fig_heat, width="stretch")

        # ── Grid 2×2 de mapas de calor — los 4 algoritmos ─────────────────
        st.divider()
        st.markdown("#### Comparativa de mapas de calor — los 4 algoritmos")
        st.caption("Cada mapa muestra la geometría de búsqueda característica del algoritmo.")

        heat_cols = st.columns(2)
        badge_colors = {"BFS": "#06B6D4", "DFS": "#F43F5E", "Greedy": "#10B981", "A*": "#6366F1"}
        for h_idx, r in enumerate(all_results):
            with heat_cols[h_idx % 2]:
                aname = r["Algorithm"]
                col = badge_colors.get(aname, "#888")
                st.markdown(
                    f'<span class="algo-badge" style="background:{col}22;color:{col};border:1px solid {col}66">{aname}</span>',
                    unsafe_allow_html=True,
                )
                fig_h = heatmap_maze(
                    maze,
                    r["explored"],
                    path=r["path"],
                    algo_name=aname,
                )
                st.pyplot(fig_h, width="stretch")

    # ══════════════════════════════════════════════
    # TAB 3 — Comparación entre algoritmos
    # ══════════════════════════════════════════════
    with tab3:
        st.markdown("### Comparación de los 4 algoritmos")
        st.caption(f"Heurística usada para Greedy y A*: **{st.session_state.get('heuristic', heuristic_name)}**")

        # DataFrame para tabla (sin columns internas como explored/path)
        df_display = pd.DataFrame([{
            "Algoritmo":         r["Algorithm"],
            "Path Length":       r["Path Length"],
            "Nodos Explorados":  r["Nodes Explored"],
            "Runtime (s)":       r["Runtime (s)"],
            "Branching Factor":  branching_factor(r["Nodes Explored"], r["Path Length"]),
        } for r in all_results])

        # Resaltado: verde mudo para mínimo, rosado mudo para máximo (Slate-theme friendly)
        styled = df_display.style.highlight_min(
            subset=["Path Length", "Nodos Explorados", "Runtime (s)", "Branching Factor"],
            color="#064E3B",  # Emerald 900
        ).highlight_max(
            subset=["Path Length", "Nodos Explorados", "Runtime (s)", "Branching Factor"],
            color="#881337",  # Rose 900
        ).format({"Runtime (s)": "{:.6f}", "Branching Factor": "{:.4f}"})


        st.dataframe(styled, width="stretch", hide_index=True)

        # ── Botón de descarga ─────────────────────────────────────────────────
        import io, datetime
        meta_rows = [
            {"Algoritmo": "— METADATOS —",  "Path Length": "",   "Nodos Explorados": "",   "Runtime (s)": ""},
            {"Algoritmo": "Laberinto",       "Path Length": f"{maze.rows}x{maze.cols}", "Nodos Explorados": maze.rows * maze.cols, "Runtime (s)": ""},
            {"Algoritmo": "Heurística",      "Path Length": st.session_state.get("heuristic", heuristic_name), "Nodos Explorados": "", "Runtime (s)": ""},
            {"Algoritmo": "Fecha",           "Path Length": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "Nodos Explorados": "", "Runtime (s)": ""},
            {"Algoritmo": "— RESULTADOS —", "Path Length": "",   "Nodos Explorados": "",   "Runtime (s)": ""},
        ]
        df_export = pd.concat(
            [pd.DataFrame(meta_rows), df_display],
            ignore_index=True,
        )

        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="Descargar resultados (CSV)",
                data=df_export.to_csv(index=False).encode("utf-8"),
                file_name=f"maze_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary",
            )

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

    # ══════════════════════════════
    # TAB 4 — Benchmark automático
    # ══════════════════════════════
    with tab4:
        st.markdown("### Benchmark automático")
        st.caption(
            "Ejecuta múltiples experimentos con pares **(start, goal) aleatorios** "
            "sobre el mismo laberinto y calcula estadísticas agregadas por algoritmo."
        )

        bcol1, bcol2, bcol3 = st.columns([1, 1, 2])
        with bcol1:
            n_exp = st.number_input(
                "Número de experimentos", min_value=5, max_value=100,
                value=20, step=5,
                help="Cada experimento usa un par (start, goal) diferente y aleatorio.",
            )
        with bcol2:
            bench_seed = st.number_input(
                "Semilla aleatoria", min_value=0, max_value=9999,
                value=42, step=1,
                help="Cambia la semilla para obtener distintos pares aleatorios.",
            )
        with bcol3:
            st.markdown("")
            run_bench = st.button(
                "Ejecutar Benchmark", type="primary",
                help="Ejecuta los 4 algoritmos sobre N pares aleatorios.",
            )

        if run_bench:
            grid_key_b = tuple(tuple(row) for row in maze_grid)
            with st.spinner(f"Ejecutando {n_exp} experimentos por los 4 algoritmos..."):
                df_stats, df_raw, valid_n = run_benchmark(
                    grid_key_b, heuristic_name, int(n_exp), int(bench_seed)
                )
            st.session_state["bench_stats"] = df_stats
            st.session_state["bench_raw"]   = df_raw
            st.session_state["bench_valid"] = valid_n
            st.session_state["bench_n"]     = int(n_exp)

        if "bench_stats" in st.session_state:
            df_stats = st.session_state["bench_stats"]
            df_raw   = st.session_state["bench_raw"]
            valid_n  = st.session_state["bench_valid"]
            n_total  = st.session_state["bench_n"]

            st.success(f"{valid_n}/{n_total} experimentos válidos completados.")

            # ── Tabla de estadísticas ─────────────────────────────────────
            st.markdown("#### Estadísticas por algoritmo")
            styled_bench = df_stats.style.highlight_min(
                subset=["Avg Runtime (s)", "Avg Nodos", "Avg Path"],
                color="#1A4731",
            ).highlight_max(
                subset=["Avg Runtime (s)", "Avg Nodos", "Avg Path"],
                color="#4A1A1A",
            ).format({
                "Avg Runtime (s)": "{:.6f}",
                "Std Runtime":     "{:.6f}",
                "Min Runtime":     "{:.6f}",
                "Max Runtime":     "{:.6f}",
                "Avg Nodos":       "{:.1f}",
                "Std Nodos":       "{:.1f}",
                "Avg Path":        "{:.1f}",
            })
            st.dataframe(styled_bench, width="stretch", hide_index=True)

            # ── Gráficos de distribución ───────────────────────────────────
            st.divider()
            st.markdown("#### Distribución por experimento")
            import altair as alt

            _BENCH_COLORS = {
                "BFS": "#06B6D4", "DFS": "#F43F5E",
                "Greedy": "#10B981", "A*": "#6366F1",
            }
            _bench_scale = alt.Scale(
                domain=list(_BENCH_COLORS.keys()),
                range=list(_BENCH_COLORS.values()),
            )

            def _bench_chart(y_field, y_title, fmt="d"):
                return (
                    alt.Chart(df_raw)
                    .mark_line(point=True, strokeWidth=2)
                    .encode(
                        x=alt.X("Experimento:Q", axis=alt.Axis(labelColor="#94A3B8")),
                        y=alt.Y(f"{y_field}:Q", title=y_title,
                                axis=alt.Axis(labelColor="#94A3B8", titleColor="#F8FAFC")),
                        color=alt.Color("Algoritmo:N", scale=_bench_scale,
                                        legend=alt.Legend(labelColor="#94A3B8", titleColor="#F8FAFC")),
                        tooltip=["Algoritmo:N", "Experimento:Q",
                                 alt.Tooltip(f"{y_field}:Q", format=f".{fmt}")],
                    )
                    .properties(height=260)
                    .configure_view(strokeOpacity=0)
                    .configure(
                        background="transparent",
                        axis=alt.AxisConfig(gridColor="#334155", domainColor="#334155"),
                    )
                )

            ch1, ch2, ch3 = st.columns(3)
            with ch1:
                st.markdown("**Runtime por experimento**")
                st.altair_chart(_bench_chart("Runtime (s)", "Runtime (s)", "6f"), width="stretch")
            with ch2:
                st.markdown("**Nodos explorados**")
                st.altair_chart(_bench_chart("Nodos", "Nodos explorados"), width="stretch")
            with ch3:
                st.markdown("**Longitud del camino**")
                st.altair_chart(_bench_chart("Path Length", "Path length"), width="stretch")

            # ── Descarga ──────────────────────────────────────────────────
            st.divider()
            import datetime, io
            dl1, dl2, _ = st.columns([1, 1, 2])
            with dl1:
                st.download_button(
                    label="Descargar Resumen (CSV)",
                    data=df_stats.to_csv(index=False).encode("utf-8"),
                    file_name=f"benchmark_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            with dl2:
                st.download_button(
                    label="Descargar Datos Crudos (CSV)",
                    data=df_raw.to_csv(index=False).encode("utf-8"),
                    file_name=f"benchmark_raw_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
        else:
            st.info("Configura los parámetros y presiona **Ejecutar Benchmark** para comenzar.")