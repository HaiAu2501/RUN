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
from sklearn.cluster import DBSCAN

import numpy as np

def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    k = min(5, n - 1)  # Number of nearest neighbors to consider
    
    # Hyperparameters
    clustering_eps = 0.45  # Tuning based on experimental performance
    clustering_min_samples = 2  # Minimum samples for defining a cluster
    distance_exponent = 1.25   # Fine-tuned for distance weighting
    connectivity_weight = 0.55   # Adjusted weight for better connectivity consideration
    
    # Step 1: Average distance for each node
    average_distance = np.mean(distance_matrix, axis=1)
    
    # Step 2: Distance rankings and closest distances
    distance_ranking = np.argsort(distance_matrix, axis=1)
    closest_distances = distance_matrix[np.arange(n)[:, None], distance_ranking[:, 1:k + 1]]
    
    # Step 3: Perform clustering to assess connectivity
    clustering = DBSCAN(eps=clustering_eps, min_samples=clustering_min_samples).fit(distance_matrix)
    cluster_labels = clustering.labels_
    
    # Initialize edge importance matrix and connectivity scores
    edge_importance = np.zeros_like(distance_matrix)
    connectivity_scores = np.zeros_like(distance_matrix)

    # Step 4: Compute edge importance based on distance and connectivity
    for i in range(n):
        for j in distance_ranking[i, 1:k + 1]:
            connectivity_scores[i, j] += 1  # Count connections to the nearest k nodes

        normalized_connectivity = (np.log(connectivity_scores[i] + 1) / np.log(n)).clip(0, 1)

        # Combine distance effects with enhanced connectivity
        distance_factor = np.power(distance_matrix[i] / average_distance[i], distance_exponent)
        edge_importance[i] = distance_factor + connectivity_weight * normalized_connectivity ** 2.5

    # Prevent self-loops
    np.fill_diagonal(edge_importance, np.inf)
    
    # Refine guidance using nearest neighbor means for edge penalties
    closest_mean_distance = np.mean(closest_distances, axis=1)
    epsilon = np.finfo(float).eps  # Small number for numerical stability 
    edge_importance += closest_mean_distance[:, np.newaxis] / (np.sum(distance_matrix, axis=1)[:, np.newaxis] + epsilon)
    
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