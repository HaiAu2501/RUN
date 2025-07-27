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



# Func96
def heuristics(distance_matrix):
    n = distance_matrix.shape[0]  # Number of cities
    
    # Hyperparameters
    density_weight = 1.2  # Local density influence adjustment
    penalty_scaling_factor = 1.25  # Enhance penalty decisions
    usage_decay = 0.6  # Decay factor for tracking edge usage
    edge_variability_inf = 1.3  # Variability influence adjustment
    normalization_offset = 1e-10  # To prevent zero division
    
    # Initialize matrices
    importance = np.zeros((n, n))
    edge_usage_frequency = np.zeros((n, n))  # Edge usage tracker
    
    # Compute importance matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_distance = distance_matrix[i, j]
                
                # Update usage frequency with decay
                edge_usage_frequency[i, j] = (edge_usage_frequency[i, j] * usage_decay) + 1
                
                # Compute local density using a refined exponential decay
                local_density = np.sum(np.exp(-distance_matrix[i, :] / edge_distance) + np.exp(-distance_matrix[:, j] / edge_distance) - 2) 
                
                # Calculate distance importance with an enhanced polynomial influence
                distance_importance = np.exp(-edge_variability_inf * (edge_distance**2) / (1 + edge_distance))
                
                # Combine importance measures while adjusting for usage frequency
                adjusted_usage = (edge_usage_frequency[i, j] ** (1.0 / (1 + np.log(edge_usage_frequency[i, j] + normalization_offset))))
                importance[i, j] = (distance_importance * (1 + density_weight * local_density)) / (adjusted_usage + normalization_offset)  

    # Normalize and scale the importance values
    min_importance = np.min(importance)
    importance -= min_importance  # Shift to non-negative range
    max_importance = np.max(importance)
    if max_importance > 0:
        importance /= max_importance  # Normalize values to [0,1]
    
    # Apply penalty scaling to enhance decision boundaries
    guide = importance * penalty_scaling_factor
    
    return guide





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