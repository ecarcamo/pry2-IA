import heapq


def astar(maze, heuristic):

    start = maze.start
    goal = maze.goal

    open_list = []

    heapq.heappush(open_list, (0, start))

    parent = {}

    g_cost = {start: 0}

    visited = set()

    nodes_explored = 0
    explored_order = []

    while open_list:

        _, current = heapq.heappop(open_list)

        if current in visited:
            continue

        visited.add(current)

        nodes_explored += 1
        explored_order.append(current)

        if current == goal:
            break

        for neighbor in maze.get_neighbors(current):

            new_cost = g_cost[current] + 1

            if neighbor not in g_cost or new_cost < g_cost[neighbor]:

                g_cost[neighbor] = new_cost

                priority = new_cost + heuristic(neighbor, goal)

                heapq.heappush(open_list, (priority, neighbor))

                parent[neighbor] = current

    path = reconstruct_path(parent, start, goal)

    return path, nodes_explored, explored_order

def reconstruct_path(parent, start, goal):
    """Reconstruye el camino de start a goal. Retorna [] si no hay camino."""

    if goal not in parent and goal != start:
        return []

    path = []

    current = goal

    while current != start:

        path.append(current)
        current = parent[current]

    path.append(start)

    path.reverse()

    return path