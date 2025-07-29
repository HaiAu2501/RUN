import numpy as np

def edge_score(i: int, j: int, distances: np.ndarray, demands: np.ndarray, capacity: int) -> float:
    """
    Score for including edge (i,j) in CVRP tour construction.
    Higher score = better edge to include.
    Note: Node 0 is the depot, customers are nodes 1 to n-1.
    
    Parameters
    ----------
    i : int
        First node index (0 = depot, 1+ = customers)
    j : int  
        Second node index (0 = depot, 1+ = customers)
    distances : np.ndarray, shape (n, n)
        Distance matrix between all nodes (depot at index 0)
    demands : np.ndarray, shape (n,)
        Demand for each node (depot has demand 0)
    capacity : int
        Vehicle capacity constraint
        
    Returns
    -------
    float
        Score for edge (i,j). Higher score means better edge.
    """
    # Basic implementation: prefer short distances and small demands
    # Shorter edges get higher scores
    distance_score = -distances[i, j]
    
    # Prefer customers with smaller demands (easier to fit)
    demand_penalty = -demands[j] / capacity
    
    return distance_score + demand_penalty