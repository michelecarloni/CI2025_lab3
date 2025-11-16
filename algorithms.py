import numpy as np
import copy
from collections import deque
import networkx as nx
import heapq

def bfs_path(graph: nx.DiGraph, start_node: int, goal_node: int):
    """
    Breadth-First Search implementation.
    """
    if start_node == goal_node:
        return [start_node]

    # queue for the nodes to visit
    queue = deque([(start_node, [start_node])])
    # set to keep track of visited nodes to avoid cycles
    visited = {start_node}

    while queue:
        current_node, path = queue.popleft()

        # we look at all neighbors of the current node
        # the graph G already correctly handles the np.inf edges
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                if neighbor == goal_node:
                    # this is the goal
                    # when the goal is found the function returns
                    return path + [neighbor]
                
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    # if the while loop finishes, no path was found
    return None

        

def dfs_path(graph: nx.DiGraph, start_node: int, goal_node: int):
    """
    Depth-First Search algorithm implementation.
    """
    if start_node == goal_node:
        return [start_node]
    
    # stack for the nodes to visit, storing (node, path_to_node)
    stack = [(start_node, [start_node])]
    # set to keep track of visited nodes to avoid cycles
    visited = set()

    while stack:
        current_node, path = stack.pop()

        if current_node == goal_node:
            # this is the goal
            return path

        # we only mark a node visited when we pop it,
        # or we check if it's visited *before* pushing.
        if current_node not in visited:
            visited.add(current_node)

            # iterating through neighbors and add them to the stack
            # reverse the neighbors to make the search order
            for neighbor in reversed(list(graph.neighbors(current_node))):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None


def dijkstra_path(graph: nx.DiGraph, start_node: int, goal_node: int):
    """
    Dijkstra's algorithm implementation.
    """
    if start_node == goal_node:
        return [start_node], 0

    distances = {node: np.inf for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    distances[start_node] = 0
    pq = [(0, start_node)]
    visited = set()

    while pq:
        current_cost, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == goal_node:
            break
            
        for neighbor in graph.neighbors(current_node):
            if neighbor in visited:
                continue
                
            edge_weight = graph[current_node][neighbor].get('weight', 1)
            new_cost = current_cost + edge_weight

            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (new_cost, neighbor))
                
    # reconstructing the path
    if distances[goal_node] == np.inf:
        return None, np.inf

    path = []
    current = goal_node
    while current is not None:
        path.append(current)
        current = predecessors[current]
        
    path.reverse() 

    if path[0] == start_node:
        return path, distances[goal_node]
    else:
        return None, np.inf
    

def a_star_path(graph: nx.DiGraph, start_node: int, goal_node: int, map_coords: np.ndarray):
    """
    A* algorithm implementation.
    """
    
    def heuristic(node1, node2, map_coords):
        """
        this is the heuristic calculated as the Euclidean Distance
        """
        dist = np.sqrt(
            np.square(map_coords[node1, 0] - map_coords[node2, 0]) + 
            np.square(map_coords[node1, 1] - map_coords[node2, 1])
        )
        return (dist * 1_000).round()

    if start_node == goal_node:
        return [start_node], 0

    # using a priority queue, storing (f_cost, g_cost, node)
    # f_cost = g_cost + h_cost
    g_costs = {node: np.inf for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    
    g_costs[start_node] = 0
    h_cost_start = heuristic(start_node, goal_node, map_coords)
    
    pq = [(h_cost_start, 0, start_node)]
    
    visited = set()

    while pq:
        # get the node with the lowest f_cost
        f_cost, g_cost, current_node = heapq.heappop(pq)
        
        # we use a visited set just like Dijkstra
        if current_node in visited:
            continue
        visited.add(current_node)

        # goal found
        if current_node == goal_node:
            # Reconstruction of the path
            path = []
            current = goal_node
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            return path, g_cost

        for neighbor in graph.neighbors(current_node):
            if neighbor in visited:
                continue
                
            edge_weight = graph[current_node][neighbor].get('weight', 1)
            new_g_cost = g_cost + edge_weight

            # if we found a cheaper *actual* path (g_cost)
            if new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                predecessors[neighbor] = current_node
                
                # calculate the new f_cost for the neighbor
                h_cost = heuristic(neighbor, goal_node, map_coords)
                new_f_cost = new_g_cost + h_cost
                heapq.heappush(pq, (new_f_cost, new_g_cost, neighbor))
                
    return None, np.inf