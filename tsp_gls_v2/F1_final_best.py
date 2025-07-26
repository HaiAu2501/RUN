# Final round optimized implementation for generate_guide_matrix
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])  # Handle edge case for empty matrix

    # Hyperparameters
    history_weight = 1.5  # Weight for historical edge usage, reduced for robustness
    connectivity_weight = 2.5  # Increased importance for connectivity
    penalty_scale = 3.0  # Scale for penalty sensitivity, increased for accentuated penalties
    distance_scale = 1.5  # Sensitivity factor for normalized distances

    # Prevent division by zero and initialize edge importance matrix
    max_distance = np.nanmax(distance_matrix, initial=1)  # Prevent division by zero
    normalized_distances = distance_matrix / max_distance

    # Compute average distance and cumulative distance factors
    average_distance = np.nanmean(normalized_distances)
    distance_deviation = np.maximum(0, normalized_distances - average_distance)

    # Improved penalty scores
    penalty_scores = penalty_scale * np.square(distance_deviation) * np.exp(-distance_deviation)

    # Calculate distance scores emphasizing the significance of various edge impacts
    distance_scores = (normalized_distances ** distance_scale) * (1 - np.clip(distance_deviation, 0, None))

    # Assess connectivity and compute adjustments
    connectivity_matrix = np.sum(np.isfinite(distance_matrix), axis=1)  # Count finite connections
    connectivity_scores = connectivity_matrix / (np.max(connectivity_matrix, initial=1) + 1e-10)  # Prevent zero division

    # Adjust penalty based on connectivity dynamics and improve edge resilience
    connection_adjusted_penalty = penalty_scores * (connectivity_scores ** 2)

    # Aggregate edge importance scores with refined metrics
    edge_importance = (history_weight * distance_deviation + distance_scores * connectivity_weight + connection_adjusted_penalty)

    # Normalize to ensure non-negative manageable values
    edge_importance = np.nan_to_num(edge_importance, nan=0.0)  # Ensure no NaN values remain
    if np.max(edge_importance) > 0:
        edge_importance -= np.min(edge_importance)  # Shift to non-negative
        edge_importance /= (np.max(edge_importance) + 1e-10)  # Normalize output

    return edge_importance