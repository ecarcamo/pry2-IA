def load_maze(file_path):
    maze = []

    with open(file_path, "r") as f:
        for line in f:
            row = [int(c) for c in line.strip()]
            maze.append(row)

    return maze

def find_start_goal(maze):

    start = None
    goal = None

    for i, row in enumerate(maze):
        for j, value in enumerate(row):

            if value == 2:
                start = (i, j)

            if value == 3:
                goal = (i, j)

    return start, goal