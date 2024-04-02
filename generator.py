import random
import numpy as np

class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = []

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices[vertex] = []

    def add_edge(self, vertex1, vertex2, weight):
        self.vertices[vertex1].append((vertex2, weight))
        self.vertices[vertex2].append((vertex1, weight))
        self.edges.append((weight, vertex1, vertex2))

def kruskal_maze(rows, cols):
    graph = Graph()

    # Add all vertices to the graph
    for i in range(rows):
        for j in range(cols):
            graph.add_vertex((i, j))

    # Generate all possible horizontal edges
    for i in range(rows):
        for j in range(cols - 1):
            graph.add_edge((i, j), (i, j + 1), random.randint(1, 100))

    # Generate all possible vertical edges
    for i in range(rows - 1):
        for j in range(cols):
            graph.add_edge((i, j), (i + 1, j), random.randint(1, 100))

    # Sort edges by weight
    graph.edges.sort()

    # Initialize sets for disjoint sets
    sets = {vertex: {vertex} for vertex in graph.vertices}

    maze = []

    for weight, vertex1, vertex2 in graph.edges:
        set1 = sets[vertex1]
        set2 = sets[vertex2]

        if set1 != set2:
            # If vertices are in different sets, merge them and add the edge to the maze
            maze.append((vertex1, vertex2))
            new_set = set1.union(set2)
            for vertex in new_set:
                sets[vertex] = new_set

    return maze

def generate_matrix(maze, rows, cols):
    
    maze_cells = [['X' for _ in range(2 * cols + 1)] for _ in range(2 * rows + 1)]
    
    for (v1, v2) in maze:
        row1, col1 = v1
        row2, col2 = v2
        maze_cells[2*row1+1][2*col1+1] = ' '
        maze_cells[row1+row2+1][col1+col2+1] = ' '
    
    matrix = np.zeros((2 * rows + 1, 2 * cols + 1))

    for i in range (len(maze_cells)):
        row = maze_cells[i]
        for j in range(len(row)):
            element = row[j]
            if element == 'X':
                matrix[i][j] = 8
    
    return matrix
