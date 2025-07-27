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

import numpy as np
import random
import math
import scipy
import torch
def heuristics(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray, 
                             max_edge_usage: float = 391.1203873280028, scaling_factor: float = 0.9007942281521215, 
                             adjustment_scale: float = 0.09131254072849153) -> np.ndarray:
    """
    Enhanced edge distance adjustment combining dynamic scaling, historical data, 
    and contextual exploration-exploitation strategies.
    """
    num_nodes = edge_distance.shape[0]
    updated_edge_distance = np.copy(edge_distance)

    max_usage_cap = np.max(edge_n_used) if np.max(edge_n_used) > 0 else 1
    avg_edge_usage = np.mean(edge_n_used)

    for i in range(num_nodes):
        current_node = local_opt_tour[i]
        next_node = local_opt_tour[(i + 1) % num_nodes]

        current_usage = edge_n_used[current_node, next_node]
        usage_effect = current_usage / max_usage_cap
        
        # Adaptive scaling influenced by historical data and current usage
        adjustment = ((1 - usage_effect) * (max_edge_usage - current_usage) / max_edge_usage) if current_usage < max_edge_usage else -usage_effect * adjustment_scale
        updated_distance = edge_distance[current_node, next_node] * (scaling_factor + (1 - scaling_factor) * usage_effect) + adjustment
        
        # Ensure non-negativity and consider average usage for moderation
        updated_distance = max(updated_distance, 0)
        if avg_edge_usage > current_usage:
            moderation_factor = (avg_edge_usage - current_usage) / avg_edge_usage
            updated_distance *= (1 - moderation_factor * adjustment_scale)

        # Symmetric update for undirected graph
        updated_edge_distance[current_node, next_node] = updated_distance
        updated_edge_distance[next_node, current_node] = updated_distance

    return updated_edge_distance






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