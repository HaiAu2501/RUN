# Final round optimized implementation for edge_score
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def edge_score(i: int, j: int, distances: np.ndarray) -> float:
    # Hyperparameters
    centrality_factor = 0.5              # Greater importance to centrality influence
    neighbor_bonus_factor = 2.0           # Increased bonus for connecting lesser-connected cities
    min_distance_threshold = 1e-6          # Threshold to avoid negligible distances

    # Base score emphasizing shorter distances
    base_score = -distances[i, j] if distances[i, j] > min_distance_threshold else 0.0

    # Centrality scoring enhanced to average over multiple central reference points
    center_points = [0, distances.shape[0] // 2, distances.shape[0] - 1, 1, distances.shape[0] - 2]
    centrality_score = sum(distances[i, cp] + distances[j, cp] for cp in center_points) * centrality_factor / len(center_points)
    
    # Degree of connectivity calculation with more sensitivity
    degree_i = np.count_nonzero(distances[i] > 0)
    degree_j = np.count_nonzero(distances[j] > 0)
    avg_degree = (degree_i + degree_j) / 2

    # Connection bonus for nodes with fewer than three connections
    neighbor_bonus = neighbor_bonus_factor * (2 - (degree_i < 3) - (degree_j < 3))  # reward for lesser-connected nodes

    # Final score that integrates centrality, bonuses, and distance
    final_score = base_score + neighbor_bonus - (1.0 / (1 + centrality_score))
    return final_score