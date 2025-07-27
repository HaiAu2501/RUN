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



# Func99
def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    guide_matrix = np.zeros((n, n))

    # Hyperparameters
    max_neighbors = 7  # Increased for better averaging
    penalty_factor = 1.5  # Reasonable factor for penalty scaling
    connectivity_weight = 0.6  # Favoring connections base on proximity

    # Global metrics
    valid_distances = distance_matrix[distance_matrix > 0]
    global_avg_distance = np.mean(valid_distances)
    global_std_distance = np.std(valid_distances)

    for i in range(n):
        distances = distance_matrix[i]
        neighbor_indices = np.argsort(distances)[:max_neighbors]  # Get closest neighbors
        relevant_distances = distances[neighbor_indices]
        avg_neighbor_distance = np.mean(relevant_distances)  # Mean of neighbors' distances

        for j in range(n):
            if i != j:
                edge_distance = distance_matrix[i, j]
                importance_value = edge_distance

                # Penalty refinements based on local and global metrics
                if edge_distance > avg_neighbor_distance * penalty_factor:
                    importance_value *= 1.8  # Strong penalty for long edges
                elif edge_distance < avg_neighbor_distance:
                    importance_value *= connectivity_weight  # Encourage shorter edges

                # Context-aware scaling using global stats
                if edge_distance < global_avg_distance - global_std_distance:
                    importance_value *= 0.8  # Encourage local edges
                elif edge_distance > global_avg_distance + global_std_distance:
                    importance_value *= 1.6  # Strong penalty for distant edges

                guide_matrix[i, j] = importance_value

    # Normalize for consistent importance range
    max_val = np.max(guide_matrix)
    if max_val > 0:
        guide_matrix /= max_val

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