class Maze:

    MOVES = [
        (-1, 0),  # arriba
        (0, 1),   # derecha
        (1, 0),   # abajo
        (0, -1)   # izquierda
    ]

    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal

        self.rows = len(grid)
        self.cols = len(grid[0])


    def is_valid(self, position):

        r, c = position

        return (
            0 <= r < self.rows and
            0 <= c < self.cols and
            self.grid[r][c] != 1
        )


    def get_neighbors(self, position):

        neighbors = []

        r, c = position

        for dr, dc in self.MOVES:

            nr = r + dr
            nc = c + dc

            new_pos = (nr, nc)

            if self.is_valid(new_pos):
                neighbors.append(new_pos)

        return neighbors