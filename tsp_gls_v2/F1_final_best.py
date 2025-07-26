# Final round optimized implementation for generate_guide_matrix
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    edge_importance = np.zeros((n, n))

    # Hyperparameters for scaling
    scaling_factor = 100
    alpha_distance = 0.5  # Adjusted weight for distance impact
    beta_frequency = 0.5   # Adjusted weight for edge usage frequency

    # Calculate average distance and its scaled variance
    valid_distances = distance_matrix[distance_matrix != 0]
    average_distance = np.mean(valid_distances) if valid_distances.size else 1
    distance_variance = np.var(valid_distances) if valid_distances.size else 0

    # Compute edge importance
    for i in range(n):
        for j in range(n):
            if i != j:
                # Penalty considering average distance and variance
                distance_penalty = (distance_matrix[i, j] / (average_distance + 1e-10)) * (1 + distance_variance/100)
                edge_importance[i, j] = distance_penalty * alpha_distance
                
                # Usage frequency based on potential edges in the tour
                usage_frequency = np.count_nonzero(distance_matrix[j]) / (n - 1)
                edge_importance[i, j] += (1 - usage_frequency) * beta_frequency

    # Normalize edge importance to [0, 1] more robustly
    max_importance = np.max(edge_importance)
    if max_importance > 0:
        edge_importance /= max_importance  # Normalize to [0, 1] 

    return edge_importance * scaling_factor  # Scale to [0, 100]