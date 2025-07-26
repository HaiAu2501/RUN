# Final round optimized implementation for generate_guide_matrix
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]  # Number of cities
    np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances
    
    # Hyperparameters
    distance_penalty_weight = 0.7  # Weighting for distance influence
    connectivity_weight = 0.3  # Weighting for connectivity influence
    
    # Step 1: Calculate connectivity strength as a ratio of edge lengths
    connectivity_strength = np.zeros((n, n))
    total_distances = np.sum(np.where(distance_matrix < np.inf, distance_matrix, 0), axis=1)
    for i in range(n):
        for j in range(n):
            if i != j:
                ratio = distance_matrix[i, j] / (total_distances[i] + 1e-10)
                connectivity_strength[i, j] = 1 - ratio  # Closer distances yield higher scores
    
    # Step 2: Calculate edge importance using inverse distance and connectivity
    edge_importance = (1 / (1 + distance_matrix)) * (1 + connectivity_weight * connectivity_strength) * distance_penalty_weight
    
    # Step 3: Normalize the edge importance scores
    min_val = np.min(edge_importance)
    max_val = np.max(edge_importance)
    
    if max_val - min_val > 1e-10:
        normalized_scores = (edge_importance - min_val) / (max_val - min_val)
    else:
        normalized_scores = np.zeros_like(edge_importance)
    
    # Step 4: Invert the scores for penalty guidance
    guide = 1 - normalized_scores  # Higher value denotes edges to avoid (penalty focus)
    return guide