# Final round optimized implementation for generate_criticality_matrix
# Strategy ID: F3
# Phase: Final round (system-aware)

import numpy as np

def generate_criticality_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate edge criticality matrix for adaptive penalty increments.
    
    This strategy determines how quickly penalties accumulate by:
    - Analyzing edge characteristics to set appropriate learning rates
    - Creating differential penalty increment values for different edges
    - Balancing penalty growth to maintain effective search dynamics
    
    Parameters
    ----------
    distance_matrix : np.ndarray, shape (n, n)
        Matrix of pairwise distances between cities.
        
    Returns
    -------
    np.ndarray, shape (n, n)
        Criticality matrix where criticality[i,j] represents the penalty
        increment value for edge (i,j). Higher values mean faster penalty
        accumulation for more critical edges.
    """
    # Define hyperparameters for penalty adjustments
    alpha = 0.5  # Scaling factor for distance adjustment
    beta = 0.1   # Base penalty increment
    min_penalty = 0.05  # Minimum penalty to avoid zeroing criticality
    scaling = 2.0  # Scaling for sharper penalty growth
    threshold = np.mean(distance_matrix)
    n = distance_matrix.shape[0]
    criticality_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                normalized_distance = (distance_matrix[i, j] / threshold)
                density_factor = np.clip(np.log1p(np.sum(distance_matrix[i] <= threshold + 1)), 0, 1)  # Logarithmic to compress densities
                crit_value = beta + alpha * (1 - normalized_distance) ** scaling
                crit_value = (1 + density_factor) * crit_value  # Adjust based on density
                criticality_matrix[i, j] = max(crit_value, min_penalty)
                
    return criticality_matrix
