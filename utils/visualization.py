import numpy as np
import matplotlib.pyplot as plt
import time


def visualize_maze(maze, path=None):

    grid = np.array(maze.grid)

    display = grid.copy()

    if path:
        for r, c in path:
            display[r][c] = 4

    start_r, start_c = maze.start
    goal_r, goal_c = maze.goal

    display[start_r][start_c] = 2
    display[goal_r][goal_c] = 3

    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(display)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def animate_solution(maze, path, container):

    import numpy as np
    import matplotlib.pyplot as plt
    import time

    grid = np.array(maze.grid)

    display = grid.copy()

    start_r, start_c = maze.start
    goal_r, goal_c = maze.goal

    display[start_r][start_c] = 2
    display[goal_r][goal_c] = 3

    for step in path:

        r, c = step
        display[r][c] = 4

        fig, ax = plt.subplots(figsize=(6,6))

        ax.imshow(display)
        ax.set_xticks([])
        ax.set_yticks([])

        container.pyplot(fig)

        time.sleep(0.02)