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

# Func106
def heuristics(distance_matrix):
    # Hyperparameters
    distance_weight = 1.0  # Weight for distance scaling
    connectivity_weight = 2.5  # Weight for edge connectivity
    clustering_size = 5  # Number of closest edges considered for clustering
    normalization_factor = 1e-10  # Prevent division by zero
    threshold_factor = 1.5  # Threshold factor for neighbor inclusivity

    n = distance_matrix.shape[0]  
    guide_matrix = np.zeros((n, n))  
    np.fill_diagonal(guide_matrix, np.inf)  

    # Calculate reachability based on distance thresholds  
    reachability = np.zeros((n, n))  
    for i in range(n):  
        for j in range(n):  
            if i != j:
                neighbor_count = np.sum(distance_matrix[i] < (distance_matrix[i, j] * threshold_factor))
                reachability[i, j] = neighbor_count / (n - 1)  

    # Calculate edge importance scores for better routing decisions  
    for i in range(n):  
        for j in range(n):  
            if i != j:  
                normalized_distance = (distance_matrix[i, j] - np.min(distance_matrix[i])) / (np.mean(distance_matrix[i]) + normalization_factor)  
                connectivity_score = reachability[i, j] * connectivity_weight  
                closest_neighbors = np.partition(distance_matrix[j], clustering_size)[:clustering_size]  
                avg_closest_distance = np.mean(closest_neighbors) / (np.sum(distance_matrix[j]) + normalization_factor)  
                guide_matrix[i, j] = (normalized_distance * distance_weight) + connectivity_score + (avg_closest_distance * 0.5)
                
                # Adjust bias scaling for favorable avg distances
                if avg_closest_distance < np.mean(distance_matrix[j]):
                    guide_matrix[i, j] *= 0.65  

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