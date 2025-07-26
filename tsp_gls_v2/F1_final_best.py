# Final round optimized implementation for generate_guide_matrix
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate edge importance matrix for intelligent penalty guidance.
    
    This strategy creates the foundation for identifying problematic edges by:
    - Analyzing distance patterns to determine edge criticality
    - Setting importance values that guide penalty selection mechanisms
    - Balancing edge characteristics for effective search diversification
    
    Parameters
    ----------
    distance_matrix : np.ndarray, shape (n, n)
        Matrix of pairwise distances between cities.

    Returns
    -------
    np.ndarray, shape (n, n)
        Edge importance matrix where guide[i,j] represents the significance
        of edge (i,j) for penalty selection. Higher values indicate edges
        more likely to be targeted during search diversification.
    """
    # Hyperparameters
    epsilon = 1e-6  # Small value to avoid division by zero

    # Step 1: Calculate basic edge penalty based on distance
    penalty_matrix = np.maximum(distance_matrix, 0)

    # Step 2: Logarithmic scaling for edge penalties to emphasize larger distances
    penalty_matrix = np.log1p(penalty_matrix)  # log(1+x)

    # Step 3: Calculate degree for each city
    city_degrees = np.sum(penalty_matrix > 0, axis=1)
    normalized_degrees = city_degrees / (np.max(city_degrees) + epsilon)

    # Step 4: Inverse degree importance where less connectivity means more importance
    inv_degree_matrix = np.diag(1 / (normalized_degrees + epsilon))  

    # Step 5: Combine penalties and connectivity
    edge_importance = np.dot(inv_degree_matrix, penalty_matrix)

    # Step 6: Introduce a dynamic distance bias based on distance deviation from mean
    avg_distance = np.mean(penalty_matrix[penalty_matrix > 0])
    distance_bias = 1 + np.exp(-penalty_matrix / avg_distance)
    edge_importance *= distance_bias

    # Step 7: Normalize final importance matrix to [0,1]
    max_importance = np.max(edge_importance) + epsilon
    edge_importance = edge_importance / max_importance

    return edge_importance