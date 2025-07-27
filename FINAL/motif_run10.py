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



# Func101
def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    guide_matrix = np.zeros((n, n))  # Edge importance matrix
    edge_frequency = np.zeros((n, n))  # Track edge usage frequencies

    # Initialize hyperparameters
    alpha = 0.9  # Weight for distance
    beta = 0.05    # Weight for frequency
    gamma = 0.01   # Variance factor for penalty
    decay_factor = 0.85  # Preserve some edge usage context
    k_neighbors = 7  # Optimize for average distance calculation

    # Calculate edge importance based on distance and frequency
    for i in range(n):
        for j in range(n):
            if i != j:
                normalized_distance = distance_matrix[i, j] / (np.mean(distance_matrix[i]) + 1e-8)
                edge_frequency[i, j] *= decay_factor  # Gradual decay for older edges

                # Focused k-nearest average calculation
                distances_to_neighbors = np.partition(distance_matrix[i], k_neighbors)[:k_neighbors]
                k_nearest_average = np.mean(distances_to_neighbors) / (np.mean(distance_matrix[i]) + 1e-8)
                frequency_weight = np.power(edge_frequency[i, j], 0.5) / (np.max(edge_frequency) + 1) if np.max(edge_frequency) > 0 else 0

                # Building the guide matrix with a focus on outlier distance penalties
                guide_matrix[i, j] = (alpha * normalized_distance + beta * frequency_weight + gamma * np.clip(np.abs(normalized_distance - k_nearest_average), 0, None))
                edge_frequency[i, j] += 1  # Update edge usage count

                # Enhanced penalty for high disparity edges
                if normalized_distance > 1.3 * k_nearest_average:
                    guide_matrix[i, j] *= 2.5  # More aggressive penalty on risky edges

    # Normalize the guide matrix for consistent scaling
    min_value = np.min(guide_matrix)
    max_value = np.max(guide_matrix)
    if max_value > min_value:
        guide_matrix = (guide_matrix - min_value) / (max_value - min_value + 1e-8)**2  # Stronger normalization to enhance differences

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