from itertools import permutations
import numpy as np
from sys import maxsize
from P1_astar import AStar, compute_smoothed_traj
from utils.utils import wrapToPi

def travelling_salesman(graph, start):
    """
    Solves the traveling salesman problem for a given graph

    Inputs:
        graph: sorted distances between nodes
        start: index of the start node
    Output:
        min_path (float): the length of the shortest path
        best_path (list): the indeces of the nodes to visit to get the shortest path
    """
    # First, we get all possible permutations
    nodes = []

    for i in range(len(graph)):
        if i != start:
            nodes.append(i)
    all_permutations = permutations(nodes)

    # Next cycle through the permutations and store the path lengths
    best_path = None
    min_path = maxsize
    for perm in all_permutations:
        print(perm)
        path = 0

        # Compute path length starting at start
        k = start
        for node in perm:
            path += graph[k][node]
            k = node
        path += graph[k][start]

        # Find minimum
        if path < min_path:
            min_path = path
            best_path = perm

    # Apend the start so we go back to it
    best_path = list(best_path) + [start]

    return min_path, best_path

def generate_graph(start, objects, statespace_lo, statespace_hi, occupancy, resolution, v_des, spline_deg, spline_alpha, traj_dt):
    """
    Solves the traveling salesman problem for a given graph

    Inputs:
        start: pose of the start node
        objects: array of the object's pose

        These are just passed onto A*:
            statespace_lo, statespace_hi, occupancy, resolution

        And these are passed onto the path smoother:
            v_des, spline_deg, spline_alpha, traj_dt

    Output:
        graph (list(list)): contains the distance between all nodes
    """
    # Prep work
    astar = AStar(statespace_lo, statespace_hi, 0, 0, occupancy, resolution)
    graph = []

    # Start is the same as any other object when considering distance
    objects = [start] + objects

    # Find distances thru A*
    for i, obj in enumerate(objects):
        distances = []
        astar.set_init(obj)

        for j, obj in enumerate(objects):
            if i == j:
                distances.append(0)
            else:
                astar.set_goal(obj)
                success = astar.solve()

                if not success:
                    # If A* cannot find a path, let's assume it's because it's too long
                    distances.append(maxsize)
                else:
                    path = astar.path
                    t, _ = compute_smoothed_traj(path, v_des, spline_deg, spline_alpha, traj_dt)

                    # Estimate duration of new trajectory
                    # Ignore alignment time
                    # TODO: Make this better by using distance and alignment time
                    distances.append(t[-1])
        
        graph.append(distances)
    
    return graph

if __name__=="__main__":
    graph = [[0, 10, 15, 20], [10, 0, 35, 25],[15, 35, 0, 30], [20, 25, 30, 0]]

    start = 0
    print(travelling_salesman(graph, start))