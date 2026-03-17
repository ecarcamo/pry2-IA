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