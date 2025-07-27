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

def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    num_nodes = distance_matrix.shape[0]
    heuristic_values = np.zeros_like(distance_matrix)

    # Adaptive penalties and historical usage
    penalties = np.zeros_like(distance_matrix)

    # Iterate over each pair of nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                direct_distance = distance_matrix[i][j]
                
                # Compute minimum distance to any other node
                min_distance_i = np.min(distance_matrix[i][distance_matrix[i] > 0])
                min_distance_j = np.min(distance_matrix[j][distance_matrix[j] > 0])

                # Calculate usage-based penalty
                penalty = penalties[i][j] if penalties[i][j] > 0 else 0
                
                # Create heuristic value considering direct distance, penalties, and min distances
                heuristic_values[i][j] = max(direct_distance - (min_distance_i + min_distance_j + penalty), 0)
                
                # Increment penalty for this edge due to its cost
                penalties[i][j] += direct_distance

    # Normalizing heuristic values
    max_value = np.max(heuristic_values)
    if max_value > 0:
        heuristic_values = heuristic_values / max_value

    # Identify promising edges through clustering (simple approach)
    # Clustering may involve advanced implementations; here we use a simple approach.
    for i in range(num_nodes):
        neighbors = np.argsort(distance_matrix[i])[:num_nodes//2]  # Take closest half as neighbors
        for j in neighbors:
            if i != j:
                heuristic_values[i][j] *= 0.5  # Reduce "badness" for promising edges

    return heuristic_values







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