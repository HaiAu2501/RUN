# Final round optimized implementation for generate_criticality_matrix
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def generate_criticality_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate edge criticality matrix for adaptive penalty increments.
    
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
    n = distance_matrix.shape[0]
    
    # Hyperparameters
    alpha = 1.5  # Increased sensitivity for sharper penalty response
    beta = 0.1   # Base penalty increment
    gamma = 0.5  # Influence of edge significance

    # Initialize the criticality matrix with enhanced features
    criticality = np.zeros((n, n))
    
    # Calculate criticality based on distance and relative importance
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = distance_matrix[i, j]
                significance = np.mean(distance_matrix[i]) / (distance + 0.1)  # Avoid division by zero
                criticality[i, j] = beta + (alpha * np.exp(-distance / np.mean(distance_matrix))) + (gamma * significance)
                
    return criticality