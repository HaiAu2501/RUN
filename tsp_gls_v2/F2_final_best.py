# Final round optimized implementation for generate_factor_matrix
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def generate_factor_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    
    # Hyperparameters for adaptive penalty adjustments
    penalty_scale = 0.2  # Base penalty scaling factor
    min_penalty_factor = 0.01  # Minimum threshold for penalties
    max_penalty_factor = 3.0  # Upper limit for extreme penalties

    # Calculate the average distance while excluding self-loops
    valid_connections = distance_matrix[distance_matrix > 0]
    avg_distance = np.mean(valid_connections) if valid_connections.size > 0 else 1
    
    # Initialize the penalty matrix
    penalty_matrix = np.zeros((n, n))
    connection_density = np.sum(distance_matrix > 0, axis=1) / (n - 1)
    
    # Enhanced penalty computation with adaptive factors based on distance variance and density
    for i in range(n):
        for j in range(n):
            if i != j and distance_matrix[i, j] > 0:
                distance = distance_matrix[i, j]
                # Base penalty calculation considering density and average distance
                penalty = penalty_scale * (distance / avg_distance) * (1 + connection_density[i])
                # Non-linear adjustments for distance relative to average with sharper contrast
                if distance < avg_distance:
                    penalty *= 0.5  # Lower penalty for shorter distances
                elif distance > 1.5 * avg_distance:
                    penalty *= 2.0  # Increased penalty for longer distances
                # Clip the penalty to the defined range
                penalty_matrix[i, j] = min(max(penalty, min_penalty_factor), max_penalty_factor)
            else:
                penalty_matrix[i, j] = 0  # No penalty on self-loops or invalid edges

    return penalty_matrix