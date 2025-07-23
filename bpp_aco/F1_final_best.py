# Final round optimized implementation for initialize
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def initialize(demands: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize heuristic and pheromone matrices for Bin Packing Problem.
    
    This strategy creates the foundation for intelligent item placement by:
    - Enhancing compatibility analysis by considering not just gap sizes, but also the cumulative sum of item sizes for multi-item placements.
    - Setting initial pheromone levels to strategically guide exploration based on initial placement desirability.
    
    Parameters
    ----------
    demands : np.ndarray, shape (n,)
        Size/demand of each item to be packed.
    capacity : int
        Bin capacity constraint.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        heuristic : np.ndarray, shape (n, n)
            Matrix representing desirability of placing items together based on size compatibility.
        pheromone : np.ndarray, shape (n, n)
            Initial pheromone levels for item placement exploration.
    """
    n = len(demands)

    # Generate compatibility matrix based on combinations fitting within bin capacity
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if demands[i] + demands[j] <= capacity:
                heuristic[i, j] = 1.0  # Compatible sizes

    # Calculate gaps but also factor in combination counts for larger synergy potential
    gaps = np.clip(capacity - demands[:, None] - demands[None, :], 0, None)  
    sum_pairs = np.maximum(demands[:, None] + demands[None, :] - capacity, 0)  # Sum that exceed capacity

    # Final heuristic combining compatibility, gaps and enhancing tighter fits, discouraging overflows
    heuristic *= (1 / (1 + gaps))
    heuristic = np.where(sum_pairs > 0, 0, heuristic)  # Set negative compatibility for overflow pairs

    # Normalize heuristic to [0, 1] range
    heuristic = heuristic / heuristic.max() if heuristic.max() != 0 else heuristic

    # Improve pheromone initialization strategy
    pheromone = np.random.rand(n, n) * 0.3  # Base random level [0.0, 0.3) for a lighter exploration at start
    pheromone += np.eye(n) * 0.7  # Heavier encouragement for self-placement for better stability

    return heuristic, pheromone
