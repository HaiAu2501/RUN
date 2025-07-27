# Final round optimized implementation for initialize
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def initialize(demands: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize heuristic and pheromone matrices for Bin Packing Problem.
    
    This strategy integrates size compatibility with advanced gap analysis and dynamic pheromone adjustments to enhance placement strategies, while introducing diversity into the pheromone exploration.
    
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
            Matrix representing desirability of placing items together based on size compatibility and computed gaps.
        pheromone : np.ndarray, shape (n, n)
            Initial pheromone levels guiding item placement exploration.
    """
    n = len(demands)

    # Generate compatibility matrix based on combinations fitting within bin capacity
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if demands[i] + demands[j] <= capacity:
                heuristic[i, j] = 1.0  # Compatible sizes
                
    # Calculate gap sizes to enhance desirability for tighter fits
    gaps = np.clip(capacity - demands[:, None] - demands[None, :], 0, None)

    # Final heuristic combining compatibility and gaps while normalizing the desirability
    heuristic *= (1 / (1 + gaps))
    heuristic = heuristic / (heuristic.max() if heuristic.max() != 0 else 1)  # Normalize in [0, 1]

    # Enhanced pheromone levels with strategic randomness and decay based on heuristics
    pheromone = np.random.rand(n, n) * 0.45  # Base random level scaled down
    pheromone += np.eye(n) * 0.55  # Encourage self-placement with an increased base probability
    pheromone = np.clip(pheromone, 0, 1)  # Ensure pheromones are bounded between 0 and 1

    return heuristic, pheromone
