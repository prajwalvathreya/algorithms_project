from collections import deque
from generator import *
import random
import time

def bfs(matrix_maze, start, end):
    """
    Perform breadth-first search to find a path from start to end in the maze represented by matrix_maze.
    
    Args:
        matrix_maze (list): A matrix representation of the maze.
        start (tuple): The starting vertex.
        end (tuple): The ending vertex.
    
    Returns:
        list: A list of vertices representing the path from start to end.
    """
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

def dfs(matrix_maze, start, end):
    """
    Perform depth-first search to find a path from start to end in the maze represented by matrix_maze.

    Args:
        matrix_maze (list): A matrix representation of the maze.
        start (tuple): The starting vertex.
        end (tuple): The ending vertex.

    Returns:
        list: A list of vertices representing the path from start to end.
    """
    visited = set()
    stack = []
    path = []
    stack.append(start)

    while stack:
        current = stack.pop()

        if current == end:
            path.append(current)
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbor in get_neighbors(current):
            row, col = neighbor
            if matrix_maze[row][col] == 0:
                stack.append(neighbor)
                path.append(neighbor)

    return path

def get_neighbors(vertex):

    """
    Get neighboring vertices of a given vertex.
    
    Args:
        vertex (tuple): A tuple representing the vertex.

    Returns:
        list: A list of neighboring vertices.

    """
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

times = {
    "bfs": 0,
    "dfs": 0,
    "dijkstras": 0,
    "a_start":0
}

while True:
    # Choose random start and end points
    all_vertices = [item for sublist in maze for item in sublist]
    start = random.choice(all_vertices)
    end = random.choice(all_vertices)
    
    # Ensure start and end are different
    if start != end:
        matrix = generate_matrix(maze, rows, cols)

        start_ = time.time()
        path_bfs = bfs(matrix, start, end)
        end_ = time.time()
        times["bfs"] = end_ - start_

        start_ = time.time()
        path_dfs = dfs(matrix, start, end)
        end_ = time.time()

        times["dfs"] = end_ - start_

        # If a valid path is found, print the matrix and path, then break the loop
        if len(path_bfs) != 0:
            print(matrix)
            print("Solution Exists for BFS")
            print(path_bfs)
        #    break

        if len(path_dfs) != 0:
            #print(matrix)
            print("Solution Exists for DFS")
            print(path_dfs)
            break

# Update matrix to highlight the path
updated_matrix_bfs = matrix.copy()
updated_matrix_dfs = matrix.copy()


for i in range(len(matrix)):
    row = matrix[i]
    for j in range(len(row)):
        if (i, j) in path_bfs:
            updated_matrix_bfs[i][j] = 1  # Highlight the path
        else:
            updated_matrix_bfs[i][j] = 0

print(updated_matrix_bfs)


for i in range(len(matrix)):
    row = matrix[i]
    for j in range(len(row)):
        if (i, j) in path_dfs:
            updated_matrix_dfs[i][j] = 1  # Highlight the path
        else:
            updated_matrix_dfs[i][j] = 0

print(updated_matrix_dfs)