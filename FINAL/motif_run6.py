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

# Func117
def heuristics(distance_matrix):
    # Hyperparameters
    k_neighbors = 5  # Number of closest edges to consider for ranking
    penalty_scale = 2.0  # Scale for penalty adjustments
    distance_adjustment = 1.2  # Adjustment factor for penalty scaling

    # Step 1: Calculate average distances and prepare normalization
    average_distance = np.mean(distance_matrix, axis=1) + 1e-10  # Prevent division by zero
    normalized_distances = distance_matrix / average_distance[:, np.newaxis]
    np.fill_diagonal(normalized_distances, np.inf)  # Avoid self-loops

    # Step 2: Calculate distances for k nearest neighbors
    distance_ranking = np.argsort(distance_matrix, axis=1)
    closest_mean_distance = np.mean(distance_matrix[np.arange(distance_matrix.shape[0])[:, None],
                                                   distance_ranking[:, 1:k_neighbors + 1]], axis=1)

    # Step 3: Use logarithmic scaling for more dynamic penalty adjustments
    total_distance = np.sum(distance_matrix, axis=1) + 1e-10  # Avoid zero division
    scaled_penalties = (closest_mean_distance / total_distance) * penalty_scale * distance_adjustment

    # Step 4: Create edge importance matrix with penalties
    penalty_adjustments = np.log1p(normalized_distances) * scaled_penalties[:, np.newaxis]  

    # Step 5: Aggregate the normalized distances and penalties for final importance
    final_edge_importance = normalized_distances + penalty_adjustments
    return final_edge_importance





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