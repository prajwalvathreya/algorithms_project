from collections import deque
from generator import *
import random

def bfs(matrix_maze, start, end):

    visited = set()
    queue = deque()
    path = []
    queue.append(start)

    while queue:

        current = queue.popleft()
        
        if current == end:
            path.append(current)
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbor in get_neighbors(current):
            row, col = neighbor
            if matrix_maze[row][col] == 0:
                queue.append(neighbor)
                path.append(neighbor)

    return path

def get_neighbors(vertex):
    row, col = vertex
    neighbors = []
    if row > 0:
        neighbors.append((row - 1, col))
    if row < rows - 1:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col < cols - 1:
        neighbors.append((row, col + 1))
    return neighbors

rows = 10
cols = 10

maze = kruskal_maze(rows, cols)

while True:

    all_vertices = [item for sublist in maze for item in sublist]

    start = random.choice(all_vertices)
    end = random.choice(all_vertices)
    
    if start != end:

        matrix = generate_matrix(maze, rows, cols)
        path = bfs(matrix, start, end)
        
        if len(path) != 0:
            print(matrix)
            print("Solution Exists")
            print(path)
            break

updated_matrix = matrix.copy()

for i in range(len(matrix)):
    row = matrix[i]
    for j in range(len(row)):
        if (i, j) in path:
            updated_matrix[i][j] = 9
        else:
            updated_matrix[i][j] = 0


print(updated_matrix)