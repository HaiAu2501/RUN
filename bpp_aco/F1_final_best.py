# Final round optimized implementation for initialize
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def initialize(demands: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize heuristic and pheromone matrices for Bin Packing Problem.
    
    This strategy refines item placement by:
    - Leveraging tighter fits through density and gap analysis.
    - Setting pheromone levels that dynamically promote combinations of compatible items.
    
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
            Matrix representing desirability of placing items together based on size compatibility and effective gaps.
        pheromone : np.ndarray, shape (n, n)
            Initial pheromone levels for item placement exploration with emphasis on dynamic compatibility.
    """
    n = len(demands)

    # Compatibility matrix to identify fitting pairs and their desirability based on gaps and sizes
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if demands[i] + demands[j] <= capacity:
                # Assign higher scores for tighter fits
                gap = capacity - (demands[i] + demands[j])
                heuristic[i, j] = 1.0 / (1 + gap)  # Higher score for smaller gaps

    # Normalize heuristic matrix to [0, 1] range based on maximum values
    heuristic = heuristic / (heuristic.max() if heuristic.max() != 0 else 1)

    # Initialize pheromone levels with an emphasis on combination compatibility
    pheromone = np.random.rand(n, n) * 0.3 + 0.7 * np.eye(n)  # Focus on self-compatibility
    pheromone *= heuristic  # Scale pheromone levels by heuristic desirability

    return heuristic, pheromone