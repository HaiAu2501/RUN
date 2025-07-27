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

# Func115
def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    epsilon = 1e-10  # Prevent division by zero

    # Step 1: Basic statistics of distances
    mean_distance = np.mean(distance_matrix, axis=1) + epsilon
    median_distance = np.median(distance_matrix, axis=1) + epsilon
    variance_distance = np.var(distance_matrix, axis=1) + epsilon

    # Step 2: Connectivity: number of direct edges per city
    connectivity = np.sum((distance_matrix != np.inf), axis=1) + epsilon

    # Step 3: Clustering coefficients based on neighbor relationships
    clustering_coeff = np.zeros(n)
    for i in range(n):
        neighbors = np.where(distance_matrix[i] != np.inf)[0]
        k = len(neighbors)
        if k > 1:
            link_count = np.sum((distance_matrix[np.ix_(neighbors, neighbors)] != np.inf).astype(int))
            clustering_coeff[i] = (2 * link_count) / (k * (k - 1)) if link_count > 0 else 0

    # Step 4: Base importance based on distance metrics
    base_importance = distance_matrix / (mean_distance[:, np.newaxis] + median_distance[:, np.newaxis] + variance_distance[:, np.newaxis])

    # Step 5: Hybrid weights considering connectivity and clustering
    hybrid_weight = (1 - (distance_matrix / (np.max(distance_matrix) + epsilon))) * (connectivity / np.sum(connectivity))[:, np.newaxis]
    hybrid_weight *= np.clip(clustering_coeff[:, np.newaxis] / (np.max(clustering_coeff) + epsilon), 0, 1)

    # Step 6: Final edge importance combining all metrics
    edge_importance = base_importance * hybrid_weight
    edge_importance *= (1 + connectivity[:, np.newaxis] / (np.max(connectivity) + epsilon))  # Adjust for connectivity impact

    # Step 7: Normalize weights to stabilize scaling
    total_weight = 1 + connectivity + clustering_coeff + epsilon
    edge_importance /= total_weight[:, np.newaxis]  # Normalize weights

    # Set diagonal elements to np.inf to avoid self-loops
    np.fill_diagonal(edge_importance, np.inf)

    return edge_importance





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