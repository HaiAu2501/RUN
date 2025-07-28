import numpy as np

def edge_score(i: int, j: int, distances: np.ndarray) -> float:
    """
    Score for including edge (i,j) in initial tour.
    Higher score = better edge to include.
    
    Parameters
    ----------
    i : int
        First city index
    j : int  
        Second city index
    distances : np.ndarray, shape (n, n)
        Distance matrix between cities
        
    Returns
    -------
    float
        Score for edge (i,j). Higher score means better edge.
    """
    # Simple implementation: shorter edges get higher scores
    # Negate distance so shorter = higher score
    return -distances[i, j]