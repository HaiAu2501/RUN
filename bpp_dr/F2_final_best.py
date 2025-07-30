import numpy as np

def item_badness(item_idx: int, permutation: list[int], demands: np.ndarray, capacity: int) -> float:
    """
    Calculate badness score of item at item_idx in the permutation.
    Higher score = worse item placement (should be removed first).
    
    Parameters
    ----------
    item_idx : int
        Index in permutation (not item ID)
    permutation : list[int]
        Current permutation of items
    demands : np.ndarray, shape (n,)
        Size/demand of each item
    capacity : int
        Bin capacity constraint
        
    Returns
    -------
    float
        Badness score. Higher = worse item placement.
    """
    item = permutation[item_idx]
    
    # Super simple: larger items are worse (harder to place)
    return demands[item] / capacity