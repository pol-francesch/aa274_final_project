from itertools import permutations
from sys import maxsize
from planners import AStar, compute_smoothed_traj
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

    return min_path, list(best_path)

def generate_graph(start, objects, occupancy, theta):
    """
    Solves the traveling salesman problem for a given graph

    Inputs:
        start: pose of the start node
        objects: array of the object's pose
        theta: current theta

        These are just passed onto A*:
            statespace_lo, statespace_hi, occupancy, resolution

        And these are passed onto the path smoother:
            v_des, spline_deg, spline_alpha, traj_dt

    Output:
        graph (list(list)): contains the distance between all nodes
    """
    # Constants from navigator
    resolution = 0.05
    plan_horizon = 15
    om_max = 0.4

    statespace_lo = snap_to_grid(resolution, (-plan_horizon, -plan_horizon))
    statespace_hi = snap_to_grid(resolution, (plan_horizon, plan_horizon))

    v_des = 0.12
    spline_alpha = 0.15
    spline_deg = 3  # cubic spline
    traj_dt = 0.1

    # Prep work
    astar = AStar(statespace_lo, statespace_hi, [0,0], [0,0], occupancy, resolution)
    graph = []

    # Start is the same as any other object when considering distance
    objects = [start] + objects

    # Find times thru A*
    for i, obj in enumerate(objects):
        times = []
        astar.set_init(obj)

        for j, obj in enumerate(objects):
            if i == j:
                times.append(0)
            else:
                astar.set_goal(obj)
                astar.reset()
                success = astar.solve()

                if not success:
                    # If A* cannot find a path, let's assume it's because it's too long
                    times.append(maxsize)
                else:
                    path = astar.path
                    t, traj = compute_smoothed_traj(path, v_des, spline_deg, spline_alpha, traj_dt)

                    # Estimate duration of new trajectory
                    th_init_new = traj[0, 2]
                    th_err = wrapToPi(th_init_new - theta)
                    t_init_align = abs(th_err / om_max)
                    t_remaining_new = t_init_align + t[-1]
                    times.append(t_remaining_new)
        
        graph.append(times)
    
    return graph

def solve_tsp_from_map(start, objects, occupancy, theta, start_idx=0):
    graph = generate_graph(start, objects, occupancy, theta)
    path_length, path_order = travelling_salesman(graph, start_idx)

    return path_length, path_order

def snap_to_grid(resolution, x):
    return (
        resolution * round(x[0] / resolution),
        resolution * round(x[1] / resolution),
    )

if __name__=="__main__":
    graph = [[0, 10, 15, 20], [10, 0, 35, 25],[15, 35, 0, 30], [20, 25, 30, 0]]

    start = 0
    print(travelling_salesman(graph, start))