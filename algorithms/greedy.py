import heapq


def greedy(maze, heuristic):

    start = maze.start
    goal = maze.goal

    open_list = []

    heapq.heappush(open_list, (0, start))

    visited = set()
    visited.add(start)

    parent = {}

    nodes_explored = 0

    while open_list:

        _, current = heapq.heappop(open_list)

        nodes_explored += 1

        if current == goal:
            break

        for neighbor in maze.get_neighbors(current):

            if neighbor not in visited:

                visited.add(neighbor)

                parent[neighbor] = current

                priority = heuristic(neighbor, goal)

                heapq.heappush(open_list, (priority, neighbor))

    path = reconstruct_path(parent, start, goal)

    return path, nodes_explored

def reconstruct_path(parent, start, goal):

    path = []

    current = goal

    while current != start:

        path.append(current)
        current = parent[current]

    path.append(start)

    path.reverse()

    return path