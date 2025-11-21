# CI2025_lab3

This is the repository for the 3rd lab of the course Computational Intelligence at Politecnico di Torino.

## Introduction

The project focuses on solving pathfinding problems within directed graphs that are generated procedurally. These graphs simulate different topological conditions by varying specific hyperparameters. The goal is to analyze how different algorithms behave under varying constraints.

The experiments cover a comprehensive grid of hyperparameters, creating a wide range of problem instances. The specific parameters tested are:

* **Sizes:** `[10, 20, 50, 100, 200, 500, 1000]` (Number of nodes)
* **Densities:** `[0.2, 0.5, 0.8, 1.0]` (Probability of edge creation)
* **Noise Levels:** `[0.0, 0.1, 0.5, 0.8]` (Randomness added to weights)
* **Negative Values:** `[False, True]` (Whether negative edge weights are allowed)

## Algorithms

Five distinct pathfinding strategies have been implemented and tested to analyze their trade-offs between speed, optimality, and path characteristics:

* **BreadthFirst Search (BFS):** Explores the graph layer by layer. It ignores edge weights entirely and guarantees finding the path with the fewest number of hops, serving as a baseline for unweighted shortest paths.
* **Depth-First Search (DFS):** Prioritizes exploring as deep as possible into the graph before backtracking. While useful for connectivity, it is not guaranteed to find the shortest path and often produces very long, winding routes in dense graphs.
* **Dijkstra's Algorithm:** Provides the optimal shortest path in graphs with non-negative weights by greedily selecting the closest unvisited node.
* **A\* Search:** An informed search algorithm that utilizes a Euclidean distance heuristic derived from the 2D coordinates of the nodes to guide the search toward the goal more efficiently than Dijkstra.
* **Bellman-Ford:** Utilized via the `networkx` library, this algorithm is included specifically to handle graphs with negative edge weights, as it can correctly identify shortest paths where Dijkstra fails and detect negative cost cycles.

**NOTE:** Dijkstra and A* Search, because of how they work, don't guarantee to find the best possible path when edges are negative. Their implementation don't return any error if the values are negative, instead they return a possible path found which might not be optimal but still feasible. 

## Project Structure

The codebase is organized into three primary files (.ipynb), each serving a distinct stage in the experimental pipeline:

* **`algorithms.ipynb`**
    It contains the implementation of 4 of the algorithms tested BFS, DFS, Dijkstra, and A* algorithms.

* **`run_tests.ipynb`**
    It has the code to run the tests over all the possible problems. The 'raw' results are saved in the directory `results`.
    The user doesn't have to re run all the tests since the results are already available and it would be computationally expensive.

* **`query_data.ipynb`**
    This file contains the function to visualize the data computed during the tests. Use this module to verify the data of intererst.