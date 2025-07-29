import numpy as np

def insert_position(customer: int, permutation: list[int], distances: np.ndarray, 
                   demands: np.ndarray, capacity: int) -> int:
    """
    Find best position to insert customer in permutation for CVRP.
    Note: Node 0 is the depot, customers are nodes 1 to n-1.
    
    Parameters
    ----------
    customer : int
        Customer ID to insert (must be > 0, since 0 is depot)
    permutation : list[int]
        Current permutation of customers (excluding depot)
    distances : np.ndarray, shape (n, n)
        Distance matrix between all nodes (depot at index 0)
    demands : np.ndarray, shape (n,)
        Demand for each node (depot has demand 0)
    capacity : int
        Vehicle capacity constraint
        
    Returns
    -------
    int
        Best position index to insert customer (0 to len(permutation))
    """
    # Simple strategy: insert next to nearest customer
    if len(permutation) == 0:
        return 0
    
    # Find nearest customer in permutation
    nearest_idx = 0
    min_distance = distances[customer, permutation[0]]
    
    for i, perm_customer in enumerate(permutation):
        dist = distances[customer, perm_customer]
        if dist < min_distance:
            min_distance = dist
            nearest_idx = i
    
    # Insert next to nearest customer (after it)
    return nearest_idx + 1