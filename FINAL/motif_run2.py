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
    heuristics_matrix = np.zeros((n, n))

    # Calculate statistical measures of distances for each node
    avg_distance = np.mean(distance_matrix, axis=1)
    median_distance = np.median(distance_matrix, axis=1)
    skewness_distance = skew(distance_matrix, axis=1, nan_policy='omit')

    # Count edges for nodes (connectivity)
    edge_count = np.sum(distance_matrix != np.inf, axis=1)

    for i in range(n):
        for j in range(n):
            if i != j:
                edge_weight = distance_matrix[i, j]
                degree_i = edge_count[i]
                degree_j = edge_count[j]

                # Base heuristic value based on edge weight and connectivity
                heuristics_matrix[i, j] = edge_weight * (degree_i + degree_j)

                # Adaptive penalties for edges based on average and median distances
                if edge_weight > avg_distance[i]:
                    heuristics_matrix[i, j] *= 1.5
                
                if edge_weight > median_distance[i]:
                    heuristics_matrix[i, j] += (edge_weight - median_distance[i]) ** 2

                # Penalty for low connectivity edges and skewness
                if degree_i + degree_j > 0:
                    heuristics_matrix[i, j] += (1 / (degree_i + degree_j)) * 10
                
                # Incorporate skewness-based penalties dynamically
                heuristics_matrix[i, j] += np.abs(skewness_distance[i]) * 2  

                # Favor edges that contribute to a well-connected structure
                if degree_i > 1 and degree_j > 1:
                    heuristics_matrix[i, j] *= 0.9  

                # Additional clustering insight: penalize edges that promote longer routes
                heuristics_matrix[i, j] += (np.std(distance_matrix[i]) + np.std(distance_matrix[j])) * 0.5

    # Normalize to ensure heuristics values are on a consistent scale
    max_value = np.max(heuristics_matrix) if np.max(heuristics_matrix) != 0 else 1
    heuristics_matrix /= max_value

    return heuristics_matrix





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