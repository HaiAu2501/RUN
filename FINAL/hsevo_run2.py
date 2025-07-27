import os, sys
import numpy as np
from scipy.spatial import distance_matrix
from base_gls import guided_local_search, _calculate_cost

OPTIMAL = {
    "50": 5.68457994395107,
    "100": 7.778580370400294,
    "200": 10.71194600194464,
    "500": 16.499886342078646
}

perturbation_moves_map = {
    50: 50,
    100: 50,
    200: 50,
    500: 50,
}
iter_limit_map = {
    50: 2000,
    100: 2000,
    200: 2000,
    500: 2000,
}

############# FOR HEURISTICS #############

import numpy as np
from scipy.stats import skew

import numpy as np

import numpy as np

def heuristics(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray) -> np.ndarray:
    """
    Updates edge distances in a more modular and adaptable way by applying systematic adjustments
    based on usage and local tour data. 

    Parameters:
    - edge_distance: np.ndarray - A matrix of edge distances.
    - local_opt_tour: np.ndarray - A locally optimized tour as a sequence of node indices.
    - edge_n_used: np.ndarray - A matrix that keeps track of how many times each edge has been used.

    Returns:
    - np.ndarray - An updated matrix of edge distances.
    """

    def adjust_distance(current_distance: float, usage_count: int) -> float:
        """Compute the adjusted distance based on usage with a systematic approach."""
        usage_penalty = current_distance * (0.9 ** (usage_count + 1))  
        adjustment = (1.4 ** usage_count) * (1 - 0.03 * usage_count)
        return current_distance + usage_penalty + adjustment

    def update_edges(tour: np.ndarray, distance_matrix: np.ndarray, usage_matrix: np.ndarray) -> np.ndarray:
        """Perform the edge update based on the given tour."""
        for i in range(len(tour)):
            current_node = tour[i]
            next_node = tour[(i + 1) % len(tour)]
            usage_count = usage_matrix[current_node, next_node]

            current_distance = distance_matrix[current_node, next_node]
            distance_matrix[current_node, next_node] = adjust_distance(current_distance, usage_count)
            distance_matrix[next_node, current_node] = distance_matrix[current_node, next_node]  # Symmetry

        return distance_matrix
    
    updated_edge_distance = update_edges(local_opt_tour, np.copy(edge_distance), edge_n_used)
    
    return np.maximum(updated_edge_distance, 0)  # Ensure non-negative distances






##########################################

def run(size):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'dataset', f'test_TSP{size}.npy')
    dataset = np.load(path)
    n_instances = dataset.shape[0]
    total_cost = 0
    for i in range(n_instances):
        coords = dataset[i]
        dist_matrix = distance_matrix(coords, coords)
        n = dist_matrix.shape[0]
        
        # Generate heuristic matrix
        heu = heuristics(dist_matrix.copy())
        
        # Run guided local search
        best_tour = guided_local_search(
            dist_matrix, 
            heu, 
            perturbation_moves=perturbation_moves_map[size], 
            iter_limit=iter_limit_map[size]
        )
        
        # Calculate and print the cost of the best tour
        best_cost = sum(dist_matrix[best_tour[i], best_tour[(i + 1) % n]] for i in range(n))
        total_cost += best_cost
    mean_cost = total_cost / n_instances
    opt_gap = (mean_cost - OPTIMAL[str(size)]) / OPTIMAL[str(size)] * 100
    print(f"Opt. gap for TSP{size}: {opt_gap:.6f}%")

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run(size)