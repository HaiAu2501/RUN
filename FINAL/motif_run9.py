import os, sys
import numpy as np
from scipy.spatial import distance_matrix
from base_gls import guided_local_search, _calculate_cost

OPTIMAL = {
    "50": 5.68457994395107,
    "100": 7.778580370400294,
    "200": 10.71194600194464,
    "500": 16.499886342078646
}

perturbation_moves_map = {
    50: 50,
    100: 50,
    200: 50,
    500: 50,
}
iter_limit_map = {
    50: 2000,
    100: 2000,
    200: 2000,
    500: 2000,
}

############# FOR HEURISTICS #############

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree

# Func104
def heuristics(distance_matrix):
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





##########################################

def run(size):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'dataset', f'test_TSP{size}.npy')
    dataset = np.load(path)
    n_instances = dataset.shape[0]
    total_cost = 0
    for i in range(n_instances):
        coords = dataset[i]
        dist_matrix = distance_matrix(coords, coords)
        n = dist_matrix.shape[0]
        
        # Generate heuristic matrix
        heu = heuristics(dist_matrix.copy())
        
        # Run guided local search
        best_tour = guided_local_search(
            dist_matrix, 
            heu, 
            perturbation_moves=perturbation_moves_map[size], 
            iter_limit=iter_limit_map[size]
        )
        
        # Calculate and print the cost of the best tour
        best_cost = sum(dist_matrix[best_tour[i], best_tour[(i + 1) % n]] for i in range(n))
        total_cost += best_cost
    mean_cost = total_cost / n_instances
    opt_gap = (mean_cost - OPTIMAL[str(size)]) / OPTIMAL[str(size)] * 100
    print(f"Opt. gap for TSP{size}: {opt_gap:.6f}%")

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run(size)