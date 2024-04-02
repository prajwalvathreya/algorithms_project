import random
import numpy as np

class Graph:
    """Graph class to represent the maze."""
    def __init__(self):
        """Initialize the Graph object with vertices and edges."""
        self.vertices = {}
        self.edges = []

    def add_vertex(self, vertex):
        """Add a vertex to the graph."""
        if vertex not in self.vertices:
            self.vertices[vertex] = []

    def add_edge(self, vertex1, vertex2, weight):
        """Add an edge to the graph between two vertices."""
        self.vertices[vertex1].append((vertex2, weight))
        self.vertices[vertex2].append((vertex1, weight))
        self.edges.append((weight, vertex1, vertex2))

def kruskal_maze(rows, cols):
    """Generate a maze using Kruskal's algorithm."""
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
    """
    Generate a matrix representation of the maze.
    
    Args:
        maze (list): A list of edges representing the maze.
        rows (int): The number of rows in the maze.
        cols (int): The number of columns in the maze.
    
    Returns:
        np.array: A matrix representation of the maze.
    """
    
    # Initialize a matrix with walls represented as 'X'
    maze_cells = [['X' for _ in range(2 * cols + 1)] for _ in range(2 * rows + 1)]
    
    # Add paths between cells based on the maze
    for (v1, v2) in maze:
        row1, col1 = v1
        row2, col2 = v2
        maze_cells[2*row1+1][2*col1+1] = ' '  # Cell
        maze_cells[row1+row2+1][col1+col2+1] = ' '  # Path between cells
    
    # Convert maze representation to matrix
    matrix = np.zeros((2 * rows + 1, 2 * cols + 1))
    for i in range(len(maze_cells)):
        row = maze_cells[i]
        for j in range(len(row)):
            element = row[j]
            if element == 'X':
                matrix[i][j] = 8  # Represent walls as 8
    
    return matrix
