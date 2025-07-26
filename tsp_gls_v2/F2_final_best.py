# Final round optimized implementation for generate_criticality_matrix
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def generate_criticality_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    criticality_matrix = np.zeros((n, n))
    edge_traversal_count = np.zeros((n, n))  # Track edge usage counts

    # Hyperparameters for penalty adjustments
    max_penalty_value = 100.0  # Increased maximum for broader exploration
    scaling_factor = 5.0  # Adjusted scaling for better sensitivity
    decay_rate = 0.85  # Slightly increased decay to weigh historical use more
    weight_factor = 0.4  # Emphasizing distance while balancing usage impact
    noise_factor = 0.1  # Increased noise to aid diversity in penalties

    for i in range(n):
        for j in range(n):
            if i != j:
                # Normalize distances ensuring stability
                normalized_distance = 1 / (distance_matrix[i, j] + 1e-9)
                
                # Influence of usage with decay and noise
                usage_influence = (edge_traversal_count[i, j] ** decay_rate) + (np.random.rand() * noise_factor)
                
                # Combine influences for a computed penalty
                overall_influence = (1 - weight_factor) * normalized_distance + weight_factor * usage_influence

                # Compute penalty based on overall influence while limiting values
                penalty = 1 / (1 + np.exp(-scaling_factor * (normalized_distance - overall_influence)))
                criticality_matrix[i, j] = np.clip(penalty, 0, max_penalty_value)

                # Update edge usage after penalty evaluation
                edge_traversal_count[i, j] += 1

    # Normalize criticality matrix values to unified scale
    criticality_matrix /= np.max(criticality_matrix) if np.max(criticality_matrix) > 0 else 1

    return criticality_matrix