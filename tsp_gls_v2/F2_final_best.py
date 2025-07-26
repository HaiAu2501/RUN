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
    # Hyperparameters
    min_penalty = 0.1               # Minimum penalty increment
    max_penalty = 75.0               # Adjusted maximum penalty increment limit
    base_decay_rate = 0.08           # Slightly enhanced decay rate for more effective adjustment
    distance_factor = 0.03            # Factor to enhance contribution of distance
    randomness_variance = 0.03        # Moderate randomness to diversify edge pressure
    effective_usage_decay = 1.8       # Base for frequency usage adjustment with logarithmic impact

    n = distance_matrix.shape[0]
    criticality_matrix = np.zeros((n, n))
    historical_penalties = np.zeros((n, n))  # Track historical penalties
    edge_usage_count = np.zeros((n, n))  # Track edge usage frequency

    for i in range(n):
        for j in range(n):
            if i != j:
                distance = distance_matrix[i, j] + 1e-6  # Avoid division by zero
                edge_usage_count[i, j] += 1  # Increment edge usage count
                # Calculate a dynamic penalty increment based on distance and usage frequency
                penalty_increment = (max_penalty - distance) * distance_factor / (np.log(edge_usage_count[i, j] + 1) ** effective_usage_decay)
                # Update historical penalties using dynamic scaling with decay
                historical_penalties[i, j] = (1 - base_decay_rate) * historical_penalties[i, j] + penalty_increment + np.random.rand() * randomness_variance
                # Enforce limits on penalties
                criticality_matrix[i, j] = min(max_penalty, max(min_penalty, historical_penalties[i, j]))  

    # Normalize the criticality matrix based on its mean value
    avg_factor = np.mean(criticality_matrix)
    if avg_factor > 0:
        criticality_matrix /= avg_factor

    return criticality_matrix