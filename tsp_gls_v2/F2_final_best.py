# Final round optimized implementation for generate_factor_matrix
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def generate_factor_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    
    # Hyperparameters for adaptive penalty adjustments
    penalty_scale = 0.15  # Base penalty scaling factor
    min_penalty_factor = 0.01  # Minimum threshold for penalties
    max_penalty_factor = 3.0 # Upper limit for extreme penalties

    # Calculate the non-zero average distance 
    valid_connections = distance_matrix[distance_matrix > 0]
    avg_distance = np.mean(valid_connections) if valid_connections.size > 0 else 1

    # Calculate the standard deviation of distances to adapt penalties dynamically
    distance_std = np.std(valid_connections) if valid_connections.size > 0 else 1

    # Initialize the penalty matrix
    penalty_matrix = np.zeros((n, n))
    
    # Adaptive penalty computation based on relative distances
    for i in range(n):
        for j in range(n):
            if i != j and distance_matrix[i, j] > 0:
                distance = distance_matrix[i, j]
                # Base penalty calculation
                penalty = penalty_scale * (distance / avg_distance)
                # Modify penalty based on its relation to avg distance and std dev
                if distance < avg_distance:
                    penalty *= 0.5  # Lower penalty for shorter distances
                elif distance > avg_distance + distance_std:
                    penalty *= 2.0  # Increased penalty for distances significantly above average
                # Clip the penalty to the defined range
                penalty_matrix[i, j] = min(max(penalty, min_penalty_factor), max_penalty_factor)
            else:
                penalty_matrix[i, j] = 0  # No penalty on self-loops or invalid edges

    return penalty_matrix