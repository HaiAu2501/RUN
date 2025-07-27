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
    n = distance_matrix.shape[0]
    
    # Hyperparameters for combining metrics
    distance_weight = 0.5
    connectivity_weight = 0.3
    variance_weight = 0.2
    
    # Calculation of average distance for each edge
    average_distance = np.mean(distance_matrix, axis=1)
    # Avoid division by zero
    average_distance[average_distance == 0] = 1e-6
    normalized_distances = distance_matrix / average_distance[:, np.newaxis]
    
    # Focus on direct usage of edges based on connectivity
    connectivity_count = np.count_nonzero(distance_matrix < np.inf, axis=1)
    connectivity_strength = (connectivity_count + 1e-6) / np.sum(connectivity_count)
    
    # Variance in distances for local edges
    closest_indices = np.argsort(distance_matrix, axis=1)[:, 1:6]  # Using top 5 closest edges
    local_variance = np.var(distance_matrix[np.arange(n)[:, None], closest_indices], axis=1) + 1e-6
    normalized_variance = local_variance / np.sum(local_variance)
    
    # Combined importance calculation
    importance_matrix = (distance_weight * normalized_distances + 
                         connectivity_weight * connectivity_strength[:, np.newaxis] + 
                         variance_weight * normalized_variance[:, np.newaxis])
    
    # Ensure non-negative results and practical limits
    return np.clip(importance_matrix, 0, 50)  # Lower max for more aggressive penalties





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