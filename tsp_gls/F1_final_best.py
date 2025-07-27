# Final round optimized implementation for generate_guide_matrix
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    # Hyperparameters
    proximity_factor = 0.85  # Encourages stronger local connections
    base_importance = 1.0  # Default edge importance
    high_threshold_multiplier = 10.0  # More aggressive penalties on far edges
    decay_factor = 0.4  # Refined decay for effectiveness
    saturation_limit = 6.0  # Limits for edge importance extremes
    dynamic_neighbors_factor = 6  # Increased neighborhood size for clustering
    n = distance_matrix.shape[0]  
    edge_importance_matrix = np.zeros_like(distance_matrix)  
    usage_counts = np.zeros_like(distance_matrix)  

    # Calculate distance statistics (percentiles) for adaptive thresholding
    valid_distances = distance_matrix[distance_matrix < np.inf].flatten()
    percentiles = np.percentile(valid_distances, [20, 40, 60, 80])
    low_threshold, mid_low_threshold, median_threshold, high_threshold = percentiles

    # Calculate edge importance based on distance and usage
    for i in range(n):
        influential_neighbors = np.argsort(distance_matrix[i])[:dynamic_neighbors_factor]
        cluster_distance_mean = np.mean(distance_matrix[i, influential_neighbors]) if influential_neighbors.size > 0 else np.inf
        for j in range(n):
            if i != j:
                distance = distance_matrix[i, j]
                importance = base_importance
                usage_adjustment = 1 + (1 / (usage_counts[i, j] + 1))

                # Dynamic adjustment based on proximity and established thresholds
                distance_ratio = distance / cluster_distance_mean if cluster_distance_mean > 0 else 1
                if distance < median_threshold:
                    importance *= (distance_ratio) ** 2.0  # Weight to shorter edges
                else:
                    importance /= (distance_ratio) ** decay_factor  # Adjust long edges more aggressive

                # Encourage connections based on established thresholds
                if distance < low_threshold:
                    importance *= proximity_factor
                elif distance > high_threshold:
                    importance *= (high_threshold_multiplier * (high_threshold / distance) ** decay_factor)

                # Assign to edge importance and apply a limit to modify extremities
                edge_importance_matrix[i, j] = min(importance * usage_adjustment, saturation_limit)
                usage_counts[i, j] += 1  # Increment edge usage

    np.fill_diagonal(edge_importance_matrix, np.inf)  # Prevent self-loops
    return edge_importance_matrix