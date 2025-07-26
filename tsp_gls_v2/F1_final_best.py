# Final round optimized implementation for generate_guide_matrix
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    # Hyperparameters
    epsilon = 1e-6  # Small value to avoid division by zero

    # Step 1: Basic distance penalties
    penalty_matrix = np.maximum(distance_matrix, 0)  # No negative penalties

    # Step 2: Compute average and standard deviation of distances
    avg_distance = np.mean(penalty_matrix[penalty_matrix > 0])
    distance_std_dev = np.std(penalty_matrix[penalty_matrix > 0])

    # Step 3: Adjust penalties for longer distances using standard deviation
    distance_weights = np.where(penalty_matrix > avg_distance, 1 + (distance_std_dev / avg_distance), 1)
    adjusted_penalty_matrix = penalty_matrix * distance_weights

    # Step 4: Calculate edge importance using degree-based adjustment
    city_degrees = np.sum(adjusted_penalty_matrix > 0, axis=1)  # Degree count
    normalized_degrees = city_degrees / (np.max(city_degrees) + epsilon)  # Normalize the degrees
    inv_degree_matrix = np.diag(1 / (normalized_degrees + epsilon))  # Inverse degree importance

    # Step 5: Combine penalties and connectivity with enhanced focus
    edge_importance = np.dot(inv_degree_matrix, adjusted_penalty_matrix)  # Mix connectivity and penalties

    # Step 6: Normalize the final importance matrix to range [0,1]
    max_importance = np.max(edge_importance) + epsilon
    normalized_importance = edge_importance / max_importance

    return normalized_importance