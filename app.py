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
from utils.visualization import visualize_maze


st.title("Maze Solver")

uploaded_file = st.file_uploader("Upload maze file", type=["txt"])

algorithm_name = st.selectbox(
    "Select Algorithm",
    ["BFS", "DFS", "Greedy", "A*"]
)

heuristic_name = st.selectbox(
    "Select Heuristic",
    ["Manhattan", "Euclidean"]
)

animate = st.checkbox("Animate solution")

run_button = st.button("Run Solver")


if uploaded_file and run_button:

    maze_grid = []

    for line in uploaded_file:
        row = [int(c) for c in line.decode().strip()]
        maze_grid.append(row)

    start, goal = find_start_goal(maze_grid)

    maze = Maze(maze_grid, start, goal)

    if heuristic_name == "Manhattan":
        heuristic = manhattan
    else:
        heuristic = euclidean

    if algorithm_name == "BFS":
        result = measure_algorithm(bfs, maze)

    elif algorithm_name == "DFS":
        result = measure_algorithm(dfs, maze)

    elif algorithm_name == "Greedy":
        result = measure_algorithm(lambda m: greedy(m, heuristic), maze)

    elif algorithm_name == "A*":
        result = measure_algorithm(lambda m: astar(m, heuristic), maze)

    st.subheader("Maze Info")

    st.write("Maze size:", maze.rows, "x", maze.cols)
    st.write("Start:", maze.start)
    st.write("Goal:", maze.goal)
    st.write("Heuristic used:", heuristic_name)

    st.subheader("Results")

    st.write("Path length:", result["path_length"])
    st.write("Nodes explored:", result["nodes_explored"])
    st.write("Runtime:", round(result["runtime"], 5), "seconds")

    total_cells = maze.rows * maze.cols
    explored_ratio = result["nodes_explored"] / total_cells

    st.write("Explored ratio:", round(explored_ratio * 100, 2), "%")

    st.subheader("Solution")

    if animate:

        from utils.visualization import animate_solution

        animation_container = st.empty()

        animate_solution(maze, result["path"], animation_container)

    else:

        fig = visualize_maze(maze, result["path"])
        st.pyplot(fig)

    st.subheader("Algorithm Comparison")

    algorithms = {
        "BFS": lambda m: bfs(m),
        "DFS": lambda m: dfs(m),
        "Greedy": lambda m: greedy(m, heuristic),
        "A*": lambda m: astar(m, heuristic)
    }

    results = []

    for name, algo in algorithms.items():

        r = measure_algorithm(algo, maze)

        results.append({
            "Algorithm": name,
            "Path Length": r["path_length"],
            "Nodes Explored": r["nodes_explored"],
            "Runtime (s)": round(r["runtime"], 5)
        })

    df = pd.DataFrame(results)

    st.dataframe(df)