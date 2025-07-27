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



# Func94
def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    guide_matrix = np.zeros((n, n))
    epsilon = 1e-6  # To avoid division by zero

    # Precompute statistics for enhanced calculations
    overall_distance_mean = np.mean(distance_matrix)
    overall_distance_max = np.max(distance_matrix)

    # Efficient penalty computation for edge importance
    for i in range(n):
        edge_i = distance_matrix[i]
        for j in range(n):
            if i != j:
                edge_j = distance_matrix[j]
                edge_distance = distance_matrix[i, j]
                short_count = ((edge_i < edge_distance).sum() + (edge_j < edge_distance).sum() - 1)
                frequency_penalty = np.log1p(short_count)  # Logarithmic scaling

                # Calculate connectivity factor based on overall distance sums
                total_distance = np.sum(edge_i) + np.sum(edge_j)
                normalized_connectivity = total_distance / (2 * (overall_distance_mean + epsilon))

                # Implementing clustering adjustments and density factors
                distance_ratio = edge_distance / (overall_distance_mean + epsilon)
                clustering_adjustment = (1 - distance_ratio) * (overall_distance_max / (overall_distance_mean + epsilon))
                neighborhood_effect = np.sum((edge_i < np.median(edge_i)) & (edge_j < np.median(edge_j)))
                density_factor = (n - 1) / (neighborhood_effect + epsilon)  # Adjust for self-edge

                # Final edge importance value combining all aspects
                edge_importance = (frequency_penalty / (normalized_connectivity + epsilon)) * clustering_adjustment * distance_ratio * density_factor
                guide_matrix[i, j] = edge_importance

    # Normalize guide matrix values
    max_value = np.max(guide_matrix) + epsilon
    if max_value > 0:
        guide_matrix /= max_value

    return guide_matrix





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