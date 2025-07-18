import numpy as np
from F1_final_best import initialize
from F2_final_best import compute_probabilities
from F3_final_best import update_pheromone

def construct_solutions(
    distances, demands, heuristic, pheromone, capacity, n_ants, iteration, n_iterations
):
    """
    Construct solutions for all ants using ACO for CVRP.
    Optimized version for better performance.
    """
    n_nodes = distances.shape[0]
    solutions = []
    
    # Pre-compute probabilities once per iteration instead of per node selection
    probabilities = compute_probabilities(pheromone, heuristic, iteration, n_iterations)
    
    # Pre-allocate arrays for better memory management
    demands_array = np.asarray(demands)
    
    for ant in range(n_ants):
        solution = []
        current_route = [0]  # Start at depot
        current_capacity = capacity
        current_node = 0
        
        # Use numpy boolean array instead of set for faster operations
        unvisited = np.ones(n_nodes, dtype=bool)
        unvisited[0] = False  # Depot already visited
        
        while np.any(unvisited):
            # Vectorized feasibility check
            feasible_mask = unvisited & (demands_array <= current_capacity)
            feasible_indices = np.where(feasible_mask)[0]
            
            if len(feasible_indices) == 0:
                # Return to depot, start new route
                current_route.append(0)
                solution.append(current_route)
                current_route = [0]
                current_capacity = capacity
                current_node = 0
                continue
            
            # Vectorized probability extraction and normalization
            node_probs = probabilities[current_node, feasible_indices]
            total_prob = np.sum(node_probs)
            
            if total_prob == 0:
                next_node = np.random.choice(feasible_indices)
            else:
                # Normalize and select
                normalized_probs = node_probs / total_prob
                next_node = np.random.choice(feasible_indices, p=normalized_probs)
            
            # Update state
            current_route.append(next_node)
            current_capacity -= demands_array[next_node]
            current_node = next_node
            unvisited[next_node] = False
        
        # Close last route if needed
        if len(current_route) > 1:
            current_route.append(0)
            solution.append(current_route)
        
        solutions.append(solution)
    
    return solutions

def calculate_solution_cost(solution, distances):
    """
    Calculate total cost of a solution.
    Optimized with vectorized operations where possible.
    """
    total_cost = 0.0
    for route in solution:
        if len(route) > 1:
            # Vectorized cost calculation for route
            route_array = np.array(route)
            route_costs = distances[route_array[:-1], route_array[1:]]
            total_cost += np.sum(route_costs)
    return total_cost

def run_cvrp_aco(distances, demands, coords, capacity, n_ants=50, n_iterations=100, seed=0):
    """
    Run ACO for CVRP.
    Optimized version with reduced function calls and better memory usage.
    """
    # Ensure minimum distance values
    distances = np.maximum(distances, 1e-6)

    # Initialize heuristic and pheromone once
    heuristic, pheromone = initialize(distances.copy(), demands, coords, capacity)
    
    best_cost = float('inf')
    best_solution = None
    
    # Pre-allocate arrays
    costs = np.zeros(n_ants)

    np.random.seed(seed)

    list_obj = []
    
    for iteration in range(n_iterations + 1):
        # Generate solutions
        solutions = construct_solutions(
            distances, demands, heuristic, pheromone, capacity, n_ants, iteration, n_iterations
        )
        
        # Vectorized cost calculation where possible
        for i, solution in enumerate(solutions):
            costs[i] = calculate_solution_cost(solution, distances)
        
        # Update best solution
        min_idx = np.argmin(costs)
        min_cost = costs[min_idx]
        if min_cost < best_cost:
            best_cost = min_cost
            best_solution = solutions[min_idx]
        
        # Update pheromone using F3
        pheromone = update_pheromone(pheromone, solutions, costs.tolist(), iteration, n_iterations)

        if iteration % 10 == 0:
            list_obj.append(best_cost)

    return list_obj

import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from aco import run_cvrp_aco

# Problem constants
N_ANTS = 30
N_ITERATIONS = 100
CAPACITY = 50

def eval_instance(coords, demands, capacity, n_ants, n_iter, seed):
    """
    Evaluate a single CVRP instance.
    
    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Node coordinates.
    demands : np.ndarray, shape (n,)
        Node demands.
    capacity : int
        Vehicle capacity.
    n_ants : int
        Number of ants.
    n_iter : int
        Number of iterations.
    seed : int
        Random seed.
        
    Returns
    -------
    float
        Best cost found.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create distance matrix
    dist_mat = distance_matrix(coords, coords)
    
    # Run CVRP ACO
    return run_cvrp_aco(dist_mat, demands, coords, capacity, n_ants, n_iter)

def process_file(path, n_ants, n_iter):
    """
    Process a dataset file.
    
    Parameters
    ----------
    path : str
        Path to dataset file.
    n_ants : int
        Number of ants.
    n_iter : int
        Number of iterations.
        
    Returns
    -------
    np.ndarray
        Results for all instances.
    """
    # Load dataset: shape (n_instances, n_nodes, 3) where last dim is [demand, x, y]
    dataset = np.load(path)
    demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
    
    n_instances = node_positions.shape[0]
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        results.append(eval_instance(node_positions[i], demands[i], CAPACITY, n_ants, n_iter, int(seeds[i])))
    
    return np.array(results)

def run(size):
    """
    Main evaluation function.
    
    Parameters
    ----------
    mode : str
        Evaluation mode: 'train', 'val', or 'test'.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    path = os.path.join(current_dir, 'datasets', f'test_CVRP{size}.npy') 
    
    # Process all files
    total_cost = 0

    costs = process_file(path, n_ants=N_ANTS, n_iter=N_ITERATIONS)
    print(costs.mean(axis=0).tolist())

if __name__ == "__main__":
    # Get mode from command line argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(size)