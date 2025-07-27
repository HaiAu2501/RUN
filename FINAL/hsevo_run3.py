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
                             low_usage_penalty: float = 1.175241182570657, neutral_penalty: float = 1.0,
                             high_usage_penalty: float = 0.7957263726935093, high_usage_threshold: int = 30.51374671681827,
                             medium_usage_threshold: int = 14.127280377729608, min_distance: float = 0.07937539303129802,
                             stochastic_loc: float = 1.3771712024857234, stochastic_scale: float = 0.08031036442458561,
                             decay_base: float = 0.8772292958363543, decay_epsilon: float = 4.83461765712637e-06) -> np.ndarray:
    """
    A more refined version of edge distance updates for TSP that utilizes flexibility through context-sensitive 
    penalties, adaptive mechanisms, and stochastic variations to enhance the exploration of the solution space.
    """
    num_nodes = edge_distance.shape[0]
    updated_edge_distance = np.zeros_like(edge_distance)

    for i in range(num_nodes):
        current_node = local_opt_tour[i]
        next_node = local_opt_tour[(i + 1) % num_nodes]

        base_distance = edge_distance[current_node, next_node]
        usage_count = edge_n_used[current_node, next_node]

        # Context-sensitive penalties based on edge usage frequency
        if usage_count < medium_usage_threshold:
            penalty = low_usage_penalty + (medium_usage_threshold - usage_count) * 0.1  # Encourage exploration of less used edges
        elif usage_count < high_usage_threshold:
            penalty = neutral_penalty  # Neutral
        else:
            penalty = high_usage_penalty - (usage_count - high_usage_threshold) * 0.05  # Penalize heavily used edges

        # Adaptive adjustment balancing penalty and base distance
        adjustment = 1.0 + (penalty - 1.0) * 0.5
        updated_distance = max(base_distance * adjustment, min_distance)  # Avoiding zero distance

        # Stochastic element for dynamic exploration
        stochastic_factor = np.random.normal(loc=stochastic_loc, scale=stochastic_scale)
        updated_distance *= stochastic_factor

        # Updating distance in both directions
        updated_edge_distance[current_node, next_node] = updated_distance
        updated_edge_distance[next_node, current_node] = updated_distance

    # Dynamic decay for edges not used in local_opt_tour based on feedback mechanism
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and (i not in local_opt_tour or j not in local_opt_tour):
                # Feedback decay that emphasizes isolation of unused edges
                feedback_decay = edge_n_used[i, j] / (np.max(edge_n_used) + decay_epsilon)  # Normalize by maximum usage
                decay_factor = decay_base ** (1 + feedback_decay)  # Exponential decay based on feedback
                updated_edge_distance[i, j] = max(min_distance, updated_edge_distance[i, j] * decay_factor)

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