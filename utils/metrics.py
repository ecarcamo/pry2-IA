import time


def measure_algorithm(algorithm, maze):

    start_time = time.perf_counter()

    path, nodes_explored, explored = algorithm(maze)

    end_time = time.perf_counter()

    runtime = end_time - start_time

    path_length = len(path)

    return {
        "path": path,
        "path_length": path_length,
        "nodes_explored": nodes_explored,
        "explored": explored,
        "runtime": runtime
    }


def branching_factor(nodes_explored: int, depth: int) -> float:
    """
    Estima el branching factor efectivo b usando la aproximación:
        b ≈ N^(1/d)
    donde N = nodos explorados, d = profundidad de la solución (path length).

    Casos borde:
      - depth <= 1  → imposible estimar; retorna float(nodes_explored)
      - nodes_explored == 0 → retorna 0.0
    """
    if nodes_explored == 0:
        return 0.0
    if depth <= 1:
        return float(nodes_explored)
    return round(nodes_explored ** (1.0 / depth), 4)