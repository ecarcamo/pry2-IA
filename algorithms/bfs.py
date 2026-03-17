from collections import deque


def bfs(maze):

    start = maze.start
    goal = maze.goal

    queue = deque([start])

    visited = set()
    visited.add(start)

    parent = {}

    nodes_explored = 0

    while queue:

        current = queue.popleft()

        nodes_explored += 1

        if current == goal:
            break

        for neighbor in maze.get_neighbors(current):

            if neighbor not in visited:

                visited.add(neighbor)

                parent[neighbor] = current

                queue.append(neighbor)

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