from collections import deque
from generator import *
import random
import heapq
import time
import json 

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

# a star with h as manhattan distance from node to end and g as steps taken

def a_star(matrix_maze, start, end):
    a_star_heap = []
    visited = set()
    path = []
    heapq.heappush(a_star_heap, (0, start, []))

    while a_star_heap:
        g, current, path = heapq.heappop(a_star_heap)

        if current == end:
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbor in get_neighbors(current):
            row, col = neighbor
            if matrix_maze[row][col] == 0:
                h = abs(row - end[0]) + abs(col - end[1])
                heapq.heappush(a_star_heap, (g + 1 + h, neighbor, path + [neighbor]))

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

def solver():
    maze = kruskal_maze(rows, cols)


    invalid_maze_count = 0
    end_maze_solver = False
    time_for_valid_maze = time.time_ns()

    while not end_maze_solver:
        # Choose random start and end points
        all_vertices = [item for sublist in maze for item in sublist]
        start = random.choice(all_vertices)
        end = random.choice(all_vertices)
        
        # Ensure start and end are different
        if start != end:
            matrix = generate_matrix(maze, rows, cols)

            bfs_start = time.time_ns()
            path_bfs = bfs(matrix, start, end)
            bfs_end = time.time_ns()

            if end not in path_bfs:
                invalid_maze_count += 1
                continue
        
            time_for_valid_maze = time.time_ns() - time_for_valid_maze - (bfs_end - bfs_start)

            end_maze_solver = True

            print("Found valid maze after", invalid_maze_count, "invalid mazes")
            print("Time taken to find valid maze:", time_for_valid_maze, "nano seconds")

            dfs_start = time.time_ns()
            path_dfs = dfs(matrix, start, end)
            dfs_end = time.time_ns()

            a_star_start = time.time_ns()
            path_a_star = a_star(matrix, start, end)
            a_star_end = time.time_ns()
        
            print("Start:", start)
            print("End:", end)
            print("Maze:")
            print(matrix)

            # Update matrix to highlight the path
            updated_matrix_bfs = matrix.copy()
            updated_matrix_dfs = matrix.copy()
            updated_matrix_a_star = matrix.copy()

            for i in range(len(matrix)):
                row = matrix[i]
                for j in range(len(row)):
                    if (i, j) in path_bfs:
                        updated_matrix_bfs[i][j] = 1  # Highlight the path
                    else:
                        updated_matrix_bfs[i][j] = 0

            print("BFS took", bfs_end - bfs_start, "nano seconds")
            print(path_bfs)
            print("BFS path")
            print(updated_matrix_bfs)
            print()

        
            for i in range(len(matrix)):
                row = matrix[i]
                for j in range(len(row)):
                    if (i, j) in path_dfs:
                        updated_matrix_dfs[i][j] = 1  # Highlight the path
                    else:
                        updated_matrix_dfs[i][j] = 0

            print("DFS took", dfs_end - dfs_start, "nano seconds")
            print(path_dfs)
            print("DFS path")
            print(updated_matrix_dfs)

        
            for i in range(len(matrix)):
                row = matrix[i]
                for j in range(len(row)):
                    if (i, j) in path_a_star:
                        updated_matrix_a_star[i][j] = 1  # Highlight the path
                    else:
                        updated_matrix_a_star[i][j] = 0

            print("A* took", a_star_end - a_star_start, "nano seconds")
            print(path_a_star)
            print("A* path")
            print(updated_matrix_a_star)

            transposed_matrix = np.transpose(matrix)

            return [{
                "start": start,
                "end": end,
                "bfs": path_bfs,
                "dfs": path_dfs,
                "a_star": path_a_star,
                "bfs_time": bfs_end - bfs_start,
                "dfs_time": dfs_end - dfs_start,
                "a_star_time": a_star_end - a_star_start,
                "time_to_find_valid_maze": time_for_valid_maze,
                "invalid_maze_count": invalid_maze_count,
                "maze": json.dumps(transposed_matrix.tolist())
            }]

        