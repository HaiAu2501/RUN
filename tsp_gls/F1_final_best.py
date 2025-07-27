# Final round optimized implementation for generate_guide_matrix
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree


def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]  
    if n == 0:  
        return np.zeros((0, 0))  
    
    # Hyperparameters  
    cluster_count = min(10, n // 2)  
    scaling_factor = 1e-10  
    locality_weight = 0.7  
    global_weight = 0.3  
    ranking_limit = 5  
    
    # Step 1: Normalize distances using combined mean and median for robustness  
    row_means = np.mean(distance_matrix, axis=1, keepdims=True)  
    row_medians = np.median(distance_matrix, axis=1, keepdims=True)  
    normalized_distance = distance_matrix / (row_means + row_medians + scaling_factor)  
    
    # Step 2: Perform K-means clustering for robust distance evaluation  
    kmeans = KMeans(n_clusters=cluster_count, random_state=42)  
    clusters = kmeans.fit_predict(normalized_distance)  
    
    # Step 3: Calculate edge centrality using Minimum Spanning Tree  
    mst = minimum_spanning_tree(distance_matrix).toarray()  
    edge_centrality = np.sum(mst > 0, axis=0)  
    
    # Step 4: Calculate distance rankings and averages  
    distance_rankings = np.argsort(distance_matrix, axis=1)  
    closest_mean_distance = np.mean(distance_matrix[np.arange(n)[:, None], distance_rankings[:, 1:ranking_limit]], axis=1)  
    further_mean_distance = np.mean(distance_matrix[np.arange(n)[:, None], distance_rankings[:, ranking_limit:10]], axis=1)  
    
    # Step 5: Compute edge importance matrix  
    importance = (normalized_distance / (np.mean(normalized_distance, axis=1, keepdims=True) + scaling_factor)) * locality_weight  
    importance += (closest_mean_distance[:, np.newaxis] * (edge_centrality / (np.sum(edge_centrality) + scaling_factor))) * global_weight  
    density_factor = further_mean_distance[:, np.newaxis] / (np.sum(distance_matrix, axis=1, keepdims=True) + scaling_factor)  
    importance += density_factor * (locality_weight + global_weight)  
    np.fill_diagonal(importance, np.inf)  
    
    return importance