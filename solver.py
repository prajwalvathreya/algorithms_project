from collections import deque
from generator import *
import random

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


def dijkstra(matrix_maze, start, end, unvisited_vertices):
    """
    Perform Dijkstra's algorithm to find a path from start to end in the maze represented by matrix_maze.

    Args:
        matrix_maze (list): A matrix representation of the maze.
        start (tuple): The starting vertex.
        end (tuple): The ending vertex.

    Returns:
        list: A list of vertices representing the path from start to end.
    """
    if (end not in unvisited_vertices) or (start not in unvisited_vertices):
        print("No path found.")
        # return a list with only starting vertex if no path is found
        return [start]
    # set to track unvisited vertices
    unvisited = set(unvisited_vertices)
    # dijkstra table to store shortest distance and previous vertex
    dijkstra_table = {vertex: (float('inf'), None) for vertex in unvisited}
    # use (-100, -100) for the previous vertex of the start vertex
    dijkstra_table[start] = (0, (-100, -100))
    # check if start is a wall
    if matrix_maze[start[0]][start[1]] != 0:
        return []
    # set current vertex to start
    current = start
    while unvisited:
        # mark current vertex as visited
        unvisited.remove(current)
        # update distances for neighboring vertices
        for neighbor in get_neighbors(current):
            row, col = neighbor
            # check if neighbor is unvisited and not a wall
            if neighbor in unvisited and matrix_maze[row][col] == 0:
                # add distance by one
                distance = dijkstra_table[current][0] + 1
                # update dijkstra table if new distance is shorter
                if distance < dijkstra_table[neighbor][0]:
                    dijkstra_table[neighbor] = (distance, current)
                    # print("Update: ", neighbor, " from ", current, " with distance ", distance)
                    # print("After update: ", neighbor, " from ", dijkstra_table[neighbor][1], " with distance ", dijkstra_table[neighbor][0])
        # set current vertex to a new unvisited vertex with shortest distance
        if unvisited:
            current = min([vertex for vertex in unvisited], key=lambda x: dijkstra_table[x][0])

    # initialize path list and backtrack from end to start
    path = []
    backtrack = end
    # print("dijkstra_table: ", dijkstra_table)
    # keep adding vertices to path until start is reached
    while backtrack and backtrack != (-100, -100):
        path.append(backtrack)
        # print("backtrack: ", backtrack)
        # print("dijkstra_table[backtrack][1]: ", dijkstra_table[backtrack][1])
        backtrack = dijkstra_table[backtrack][1]
    # reverse path to get start to end path
    path = path[::-1]
    # return a list with only starting vertex if no path is found
    if path[0] != start:
        print("No path found.")
        return [start]
    else:
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

while True:
    # Choose random start and end points
    all_vertices = [item for sublist in maze for item in sublist]
    start = random.choice(all_vertices)
    end = random.choice(all_vertices)

    # Ensure start and end are different
    if start != end:
        matrix = generate_matrix(maze, rows, cols)
        path_bfs = bfs(matrix, start, end)
        path_dfs = dfs(matrix, start, end)
        path_dijsktra = dijkstra(matrix, start, end, all_vertices)

        # If a valid path is found, print the matrix and path, then break the loop
        if len(path_bfs) != 0:
            print(matrix)
            print("Solution Exists for BFS.")
            print(path_bfs)
            # break

        if len(path_dfs) != 0:
            #print(matrix)
            print("Solution Exists for DFS.")
            print(path_dfs)
            # break

        if len(path_dijsktra) != 0:
            #print(matrix)
            print("Solution Exists for Dijsktra.")
            print(path_dijsktra)
            break

# Update matrix to highlight the path
updated_matrix_bfs = matrix.copy()
updated_matrix_dfs = matrix.copy()
updated_matrix_dijkstra = matrix.copy()


for i in range(len(matrix)):
    row = matrix[i]
    for j in range(len(row)):
        if (i, j) in path_bfs:
            updated_matrix_bfs[i][j] = 1  # Highlight the path
        else:
            updated_matrix_bfs[i][j] = 0

print("Solution for BFS:")
print(updated_matrix_bfs)


for i in range(len(matrix)):
    row = matrix[i]
    for j in range(len(row)):
        if (i, j) in path_dfs:
            updated_matrix_dfs[i][j] = 1  # Highlight the path
        else:
            updated_matrix_dfs[i][j] = 0

print("Solution for DFS:")
print(updated_matrix_dfs)

for i in range(len(matrix)):
    row = matrix[i]
    for j in range(len(row)):
        if (i, j) in path_dijsktra:
            # highlight the path using number 1
            updated_matrix_dijkstra[i][j] = 1
        else:
            updated_matrix_dijkstra[i][j] = 0

print("Solution for Dijsktra:")
print(updated_matrix_dijkstra)