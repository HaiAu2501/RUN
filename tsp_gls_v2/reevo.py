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
    50: 30,
    100: 40,
    200: 40,
    500: 50,
}
iter_limit_map = {
    50: 1000,
    100: 2000,
    200: 2000,
    500: 1000,
}

############# FOR HEURISTICS #############

import numpy as np

def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    heuristic_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # Tracking frequency of each edge being included in a solution
    edge_frequency = np.zeros_like(distance_matrix, dtype=float)

    # Calculate cumulative distance to incorporate dynamic penalties
    cumulative_distance = np.sum(distance_matrix, axis=1)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Average distance from i considering only valid distances (not inf)
                valid_distances = distance_matrix[i][distance_matrix[i] != np.inf]
                if valid_distances.size > 0:
                    avg_distance_from_i = np.mean(valid_distances)
                    # Base heuristic adjusted by average distance
                    heuristic_matrix[i, j] = distance_matrix[i, j] / avg_distance_from_i
                
                # Introduce a penalty for long edges dynamically based on cumulative distance
                if distance_matrix[i, j] > 1.5 * avg_distance_from_i:
                    heuristic_matrix[i, j] += 1.0
                
                # Increment edge frequency (this can be from previous iterations in practice)
                edge_frequency[i, j] += 1
                
                # Consider cumulative path cost to discourage reinserting already used edges
                heuristic_matrix[i, j] += edge_frequency[i, j] * (distance_matrix[i, j] / (1 + cumulative_distance[i]))

                # If j is the starting point (assumed node 0), add a reward for keeping it
                if j == 0:
                    heuristic_matrix[i, j] *= 0.5

    return heuristic_matrix


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
        best_cost = _calculate_cost(dist_matrix, best_tour)
        total_cost += best_cost
    mean_cost = total_cost / n_instances
    opt_gap = (mean_cost - OPTIMAL[str(size)]) / OPTIMAL[str(size)] * 100
    print(f"Opt. gap for TSP{size}: {opt_gap:.6f}%")

def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run(size)