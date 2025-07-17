# Final round optimized implementation for initialize
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def initialize(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(prize)
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    if np.any(distance < 0):
        raise ValueError("Distance matrix contains negative values.")

    heuristic = np.zeros((n, n))
    pheromone = np.ones((n, n)) * 0.5  # Decreased initial pheromone level for better exploration

    for i in range(n):
        for j in range(n):
            if i != j:
                if distance[i, j] <= maxlen:
                    decay_factor = 1 - (distance[i, j] / maxlen)
                    normalized_prize = (prize[j] * decay_factor) / (distance[i, j] + 1e-6)
                    heuristic[i, j] = normalized_prize
                    pheromone[i, j] += normalized_prize * 2.0  # Increased pheromone influence
                else:
                    heuristic[i, j] = 0

    # Enhance pheromone matrix normalization with more dynamic scaling
    pheromone_sum = pheromone.sum(axis=1, keepdims=True)
    pheromone = pheromone / (pheromone_sum + 1e-6)
    
    # Boosting heuristic, allowing more focus on combined factors of prize and distance 
    heuristic = heuristic / (np.mean(heuristic, axis=1, keepdims=True) + 1e-6)
    heuristic = np.power(heuristic, 2.0)  # Adjusted power for better selection of paths

    return heuristic, pheromone