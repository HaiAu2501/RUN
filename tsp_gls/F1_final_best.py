# Final round optimized implementation for generate_guide_matrix
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    # Hyperparameters
    distance_weight = 1.0  # Weight for distance scaling
    connectivity_weight = 2.5  # Weight for edge connectivity
    clustering_size = 5  # Number of closest edges considered for clustering
    normalization_factor = 1e-10  # Prevent division by zero
    threshold_factor = 1.5  # Threshold factor for neighbor inclusivity

    n = distance_matrix.shape[0]  
    guide_matrix = np.zeros((n, n))  
    np.fill_diagonal(guide_matrix, np.inf)  

    # Calculate reachability based on distance thresholds  
    reachability = np.zeros((n, n))  
    for i in range(n):  
        for j in range(n):  
            if i != j:
                neighbor_count = np.sum(distance_matrix[i] < (distance_matrix[i, j] * threshold_factor))
                reachability[i, j] = neighbor_count / (n - 1)  

    # Calculate edge importance scores for better routing decisions  
    for i in range(n):  
        for j in range(n):  
            if i != j:  
                normalized_distance = (distance_matrix[i, j] - np.min(distance_matrix[i])) / (np.mean(distance_matrix[i]) + normalization_factor)  
                connectivity_score = reachability[i, j] * connectivity_weight  
                closest_neighbors = np.partition(distance_matrix[j], clustering_size)[:clustering_size]  
                avg_closest_distance = np.mean(closest_neighbors) / (np.sum(distance_matrix[j]) + normalization_factor)  
                guide_matrix[i, j] = (normalized_distance * distance_weight) + connectivity_score + (avg_closest_distance * 0.5)
                
                # Adjust bias scaling for favorable avg distances
                if avg_closest_distance < np.mean(distance_matrix[j]):
                    guide_matrix[i, j] *= 0.65  

    return guide_matrix