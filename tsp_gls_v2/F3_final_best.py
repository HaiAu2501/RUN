# Final round optimized implementation for generate_criticality_matrix
# Strategy ID: F3
# Phase: Final round (system-aware)

import numpy as np

def generate_criticality_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    # Define hyperparameters for penalty adjustments
    alpha = 0.3  # Scaling factor for distance
    beta = 1.0  # Base penalty increment
    scaling = 2.0  # Non-linear scaling factor for penalties
    threshold = np.mean(distance_matrix)  # Average distance as threshold
    min_penalty = 0.1  # Minimum penalty to avoid zeroing criticality

    # Calculate criticality matrix based on distances
    n = distance_matrix.shape[0]
    criticality_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Compute normalized distance metric compared to the threshold
                normalized_distance = distance_matrix[i, j] / threshold
                # Apply a polynomial transformation for sharper penalty differentiation
                crit_value = beta + alpha * (1 - normalized_distance) ** scaling 
                crit_value = max(crit_value, min_penalty)  # Apply minimum bound
                criticality_matrix[i, j] = crit_value

    return criticality_matrix