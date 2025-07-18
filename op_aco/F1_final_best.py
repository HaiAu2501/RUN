# Final round optimized implementation for initialize
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def initialize(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(prize)
    heuristic = np.zeros((n, n))
    pheromone = np.ones((n, n)) * 1.5  # Slightly increased initial pheromone for exploration

    # Further enhanced heuristic calculation for better scaling and additional factors
    for i in range(n):
        for j in range(1, n):  # skip depot for j
            if distance[i, j] > 0 and distance[i, j] <= maxlen:
                ratio = prize[j] / distance[i, j]
                heuristic[i, j] = (ratio ** 5) * (1 + np.log1p(ratio)**3) * (np.sqrt(prize[j]) / 2)  # Increased amplification
                pheromone[i, j] += (prize[j] / (3.0 + 0.01 * maxlen))  # Higher weight for attractive nodes

    # Implementation of stronger pheromone decay and prioritization  
    pheromone *= np.exp(-1.75 * distance / (maxlen if maxlen > 0 else 1))  # Stronger decay for clear path focus
    pheromone = np.clip(pheromone, 0.5, 15.0)  # Adjusted boundaries for deeper exploration

    # Enhanced initial pheromone boosting for depot paths and extra scaling
    pheromone[0, :] *= 300.0  # Further significant increase for initial paths from depot

    # Random elements for exploration diversity now including a negative factor to allow oscillation
    pheromone += (np.random.rand(n, n) - 0.3) * 0.9  # Tighter bounds for exploration with stochastic elements

    return heuristic, pheromone