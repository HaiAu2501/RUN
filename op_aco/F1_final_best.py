# Final round optimized implementation for initialize
# Strategy ID: F1
# Phase: Final round (system-aware)

import numpy as np

def initialize(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(prize)
    heuristic = np.zeros((n, n))  
    pheromone = np.zeros((n, n))

    # Enhanced heuristic focusing on high-value, short-distance paths with improved penalty handling
    for i in range(n):
        for j in range(n):
            if i != j and distance[i, j] > 0:
                # Enhanced prize to distance calculation with adjusted scaling factor
                prize_distance_ratio = prize[j] / distance[i, j] * 1.1
                heuristic[i, j] = np.exp(prize_distance_ratio) * np.exp(-distance[i, j] / (maxlen + 1e-10))
                heuristic[i, j] = np.clip(heuristic[i, j], 0, 1e6)
                # Incremental and exponential penalty for exceeding maxlen to maintain exploration quality
                if distance[i, j] > maxlen:
                    heuristic[i, j] *= 0.03 * np.exp(-(distance[i, j] - maxlen) / (maxlen + 1e-10))

    # Robust normalization to ensure proper scaling in heuristic matrix
    heuristic_sum = np.sum(heuristic, axis=1, keepdims=True)
    heuristic = np.divide(heuristic, heuristic_sum + 1e-10, out=np.zeros_like(heuristic), where=heuristic_sum != 0)

    # Enhanced pheromone initialization increase focus on prize maximizing and distance minimizing
    for i in range(n):
        for j in range(n):
            if i != j and distance[i, j] <= maxlen:
                pheromone[i, j] = prize[j] * 1.5 * np.exp(-distance[i, j] / (7.0 + 1e-10))  # Increased tapering for relevance

    # Normalize pheromone to prevent over-saturation while retaining exploration dynamism
    pheromone_sum = np.sum(pheromone, axis=1, keepdims=True)
    pheromone = np.divide(pheromone, pheromone_sum + 1e-10, out=np.zeros_like(pheromone), where=pheromone_sum != 0)

    return heuristic, pheromone