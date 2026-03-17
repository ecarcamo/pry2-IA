from maze.loader import load_maze, find_start_goal
from maze.maze import Maze

maze_grid = load_maze("test_mazes/test_maze.txt")

start, goal = find_start_goal(maze_grid)

maze = Maze(maze_grid, start, goal)

print("Start:", maze.start)
print("Goal:", maze.goal)

neighbors = maze.get_neighbors(start)

print("Neighbors of start:", neighbors)