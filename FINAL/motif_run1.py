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
    np.random.seed(42)  # For reproducibility
    n = distance_matrix.shape[0]
    # Hyperparameters for adaptive weighting
    weight_distance = 0.5
    weight_usage = 0.3
    weight_entropy = 0.2

    # Step 1: Normalize distances to create a penalty guide
    average_distance = np.mean(distance_matrix, axis=1, keepdims=True)
    normalized_distance = distance_matrix / (average_distance + 1e-9)

    # Step 2: Calculate degree centrality as a usage measure
    degree_centrality = np.sum(distance_matrix < np.inf, axis=1)
    centrality_normalized = (degree_centrality - np.min(degree_centrality)) / (np.max(degree_centrality) - np.min(degree_centrality) + 1e-9)

    # Step 3: Implement historical usage metrics; assume simulated data captures recent usage trends
    historical_usage = np.random.rand(n, n) + 1
    total_usage = np.sum(historical_usage, axis=1)[:, np.newaxis]
    usage_ratios = historical_usage / (total_usage + 1e-9)

    # Step 4: Calculate MAD for distance consistency and uncertainty measures
    median_distance = np.median(distance_matrix, axis=1, keepdims=True)
    mad_distance = np.mean(np.abs(distance_matrix - median_distance), axis=1)[:, np.newaxis]

    # Step 5: Compute distance entropy to gauge edge selection uncertainty
    distance_entropy = -np.sum((normalized_distance / np.sum(normalized_distance, axis=1, keepdims=True)) * 
                               np.log(normalized_distance + 1e-9), axis=1)[:, np.newaxis]

    # Step 6: Create composite indicators balancing distance, usage, and diversity through entropy and MAD
    guide_matrix = (weight_distance * normalized_distance +  
                  weight_usage * usage_ratios + 
                  weight_entropy * distance_entropy + 
                  0.1 * mad_distance + 
                  0.2 * centrality_normalized[:, np.newaxis])

    # Set diagonal elements to inf to avoid self-loops
    np.fill_diagonal(guide_matrix, np.inf)

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