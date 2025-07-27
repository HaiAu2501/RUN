# Final round optimized implementation for generate_guide_matrix
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]  # Number of cities
    # Hyperparameters
    penalty_factor = 3.5  # Reasonable emphasis on long edges
    neighbor_weight = 0.5  # Influence from neighboring edges
    edge_importance = np.zeros((n, n))  # Edge scores holder
    standardization_factor = 1e-10  # Prevent division by zero

    # Calculate average and std deviation of edge lengths
    means = np.mean(distance_matrix, axis=1)
    std_devs = np.std(distance_matrix, axis=1)

    # Calculate overall edge lengths and neighborhood influences
    for i in range(n):
        for j in range(n):
            if i != j:
                # Squared length penalty for long edges
                length_penalty = (distance_matrix[i, j] ** 2) / (means[i] + standardization_factor)

                # Cumulative influence from neighbors (top 3)
                neighbor_influence = np.mean(distance_matrix[i, np.argsort(distance_matrix[i])[:3]]) / (means[i] + standardization_factor)

                # Combined penalty score
                edge_importance[i, j] = (length_penalty * penalty_factor) + (neighbor_weight * neighbor_influence)
            else:
                edge_importance[i, j] = np.inf  # Self-loop penalty

    # Normalize the edge importance matrix to enhance stability
    min_value = np.min(edge_importance[np.isfinite(edge_importance)])
    edge_importance[np.isfinite(edge_importance)] -= min_value
    max_value = np.max(edge_importance[np.isfinite(edge_importance)])
    edge_importance[np.isfinite(edge_importance)] /= (max_value if max_value > 0 else 1)

    return edge_importance