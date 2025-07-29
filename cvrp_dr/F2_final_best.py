import numpy as np

def customer_badness(customer_idx: int, permutation: list[int], distances: np.ndarray, 
                    demands: np.ndarray, capacity: int) -> float:
    """
    Calculate badness score of customer at customer_idx in the permutation.
    Higher score = worse customer (should be removed first).
    Note: Node 0 is the depot, customers are nodes 1 to n-1.
    
    Parameters
    ----------
    customer_idx : int
        Index in permutation (not customer ID)
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
    float
        Badness score. Higher = worse customer.
    """
    customer = permutation[customer_idx]
    
    # Simple badness: customers with large demands are bad
    demand_badness = demands[customer] / capacity
    
    # Add distance penalty: sum distances to all other customers in permutation
    distance_penalty = 0.0
    for other_customer in permutation:
        distance_penalty += distances[customer, other_customer]
    
    return demand_badness + distance_penalty / len(permutation)