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

# Func114
def heuristics(distance_matrix):
    n = distance_matrix.shape[0]  # number of cities
    
    # Hyperparameters
    distance_weight = 0.4  # Weight for distance impact
    variance_weight = 0.3  # Weight for edge variance impact
    connectivity_weight = 0.3  # Weight for connectivity impact
    penalty_factor = 2.0  # Factor to emphasize penalties on longer edges
    penalty_threshold = 1.5  # Threshold for applying penalties
    
    # Calculate average distances and variances
    avg_distance = np.mean(distance_matrix, axis=1)
    edge_variance = np.var(distance_matrix, axis=1)
    degree = np.sum(distance_matrix < np.inf, axis=1) - 1  # Exclude self-loops
    clustering_coeff = np.zeros(n)
    
    # calculating clustering coefficients for connectivity evaluation
    for i in range(n):
        connections = np.where(distance_matrix[i] < np.inf)[0]
        k = len(connections)
        if k > 1:
            possible_edges = k * (k - 1) / 2
            actual_edges = 0
            for j in connections:
                actual_edges += np.sum(distance_matrix[connections, j] < np.inf)
            clustering_coeff[i] = actual_edges / (2 * possible_edges)

    # Initialize edge importance matrix
    edge_importance = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate normalized distance score
                distance_score = distance_matrix[i, j] / (avg_distance[i] + 1e-10)
                # Compute contribution from edge variance
                variance_adjusted = edge_variance[j] / (edge_variance[j] + 1e-10)
                # Analyze connectivity
                connectivity_score = clustering_coeff[j]
                # Penalty logic
                if distance_score > penalty_threshold:
                    distance_score *= penalty_factor  # Emphasize penalties for long edges
                
                # Aggregate scores for edge importance
                edge_importance[i, j] = (
                    distance_weight * (1 - distance_score) + 
                    variance_weight * (1 - variance_adjusted) + 
                    connectivity_weight * (1 - connectivity_score)
                )

    # Normalize edge importance matrix
    edge_importance_normalized = (
        edge_importance - np.min(edge_importance)
    ) / (np.max(edge_importance) - np.min(edge_importance) + 1e-10)
    indicators = (1 - edge_importance_normalized)  # Invert for penalty guidance
    np.fill_diagonal(indicators, np.inf)  # Avoid self-loops
    return indicators





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