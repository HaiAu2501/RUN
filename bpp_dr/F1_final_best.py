import numpy as np

def item_compatibility(i: int, j: int, demands: np.ndarray, capacity: int) -> float:
    """
    Score for placing items i and j close together in the permutation.
    Higher score = better to place items near each other.
    
    Parameters
    ----------
    i : int
        First item index
    j : int  
        Second item index
    demands : np.ndarray, shape (n,)
        Size/demand of each item
    capacity : int
        Bin capacity constraint
        
    Returns
    -------
    float
        Compatibility score. Higher score means better to place together.
    """
    # Simple implementation: items with complementary sizes should be together
    # Items that sum to near capacity get high scores
    total_size = demands[i] + demands[j]
    
    # Prefer pairs that use capacity efficiently (close to full capacity)
    if total_size <= capacity:
        efficiency = total_size / capacity
        return efficiency  # Higher when closer to full capacity
    else:
        # Penalize pairs that exceed capacity
        return -1.0