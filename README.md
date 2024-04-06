# Maze Solver and Generator

This repository contains Python scripts for generating mazes using Kruskal's algorithm and solving them using BFS (Breadth-First Search).

## Files

- `solver.py`: Contains the implementation of BFS for solving mazes.
- `generator.py`: Contains the implementation of Kruskal's algorithm for generating mazes.

## solver.py

### Functionality

- The `bfs` function performs breadth-first search to find a path from a start to an end point in a maze represented by a matrix.
- The `get_neighbors` function retrieves neighboring vertices of a given vertex.
- The script generates a random maze using Kruskal's algorithm, then solves it using BFS, and finally highlights the solution path in the maze.

### Usage

To run the solver script, execute the following command:

```bash
python solver.py

```server
uvicorn main:app --reload