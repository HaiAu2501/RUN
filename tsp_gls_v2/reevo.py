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
from scipy.stats import skew

def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    # Hyperparameters
    proximity_factor = 0.85  # Encourages stronger local connections
    base_importance = 1.0  # Default edge importance
    high_threshold_multiplier = 10.0  # More aggressive penalties on far edges
    decay_factor = 0.4  # Refined decay for effectiveness
    saturation_limit = 6.0  # Limits for edge importance extremes
    dynamic_neighbors_factor = 6  # Increased neighborhood size for clustering
    n = distance_matrix.shape[0]  
    edge_importance_matrix = np.zeros_like(distance_matrix)  
    usage_counts = np.zeros_like(distance_matrix)  

    # Calculate distance statistics (percentiles) for adaptive thresholding
    valid_distances = distance_matrix[distance_matrix < np.inf].flatten()
    percentiles = np.percentile(valid_distances, [20, 40, 60, 80])
    low_threshold, mid_low_threshold, median_threshold, high_threshold = percentiles

    # Calculate edge importance based on distance and usage
    for i in range(n):
        influential_neighbors = np.argsort(distance_matrix[i])[:dynamic_neighbors_factor]
        cluster_distance_mean = np.mean(distance_matrix[i, influential_neighbors]) if influential_neighbors.size > 0 else np.inf
        for j in range(n):
            if i != j:
                distance = distance_matrix[i, j]
                importance = base_importance
                usage_adjustment = 1 + (1 / (usage_counts[i, j] + 1))

                # Dynamic adjustment based on proximity and established thresholds
                distance_ratio = distance / cluster_distance_mean if cluster_distance_mean > 0 else 1
                if distance < median_threshold:
                    importance *= (distance_ratio) ** 2.0  # Weight to shorter edges
                else:
                    importance /= (distance_ratio) ** decay_factor  # Adjust long edges more aggressive

                # Encourage connections based on established thresholds
                if distance < low_threshold:
                    importance *= proximity_factor
                elif distance > high_threshold:
                    importance *= (high_threshold_multiplier * (high_threshold / distance) ** decay_factor)

                # Assign to edge importance and apply a limit to modify extremities
                edge_importance_matrix[i, j] = min(importance * usage_adjustment, saturation_limit)
                usage_counts[i, j] += 1  # Increment edge usage

    np.fill_diagonal(edge_importance_matrix, np.inf)  # Prevent self-loops
    return edge_importance_matrix



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