import numpy as np

def initialize(demands: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(demands)

    # Refined heuristic calculations that prioritize fit and proximity
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and (demands[i] + demands[j] <= capacity):
                fit_score = (demands[i] + demands[j]) / capacity
                proximity_factor = (1.0 - (max(demands[i], demands[j]) / capacity)) ** 3  # Cubic factor for intense influence
                heuristic[i, j] = fit_score * proximity_factor  # Combined score

    # Improved normalization strategy for heuristic values
    heuristic_sum = heuristic.sum(axis=1, keepdims=True)
    heuristic_sum[heuristic_sum == 0] = 1  # Prevent division by zero
    heuristic = heuristic / heuristic_sum  # Normalize heuristic values

    # Enhanced pheromone initialization that applies a multiplicative boost on successful pairs
    pheromone = np.zeros_like(heuristic)
    for i in range(n):
        for j in range(n):
            if heuristic[i, j] > 0:
                pheromone[i, j] = (1 / (1 + (demands[i] + demands[j] - capacity) + 1e-6)) ** 1.5 * heuristic[i, j]  # Strong boost on promising combinations

    # Normalizing pheromone levels to ensure balanced distribution
    pheromone_sum = pheromone.sum(axis=1, keepdims=True)
    pheromone_sum[pheromone_sum == 0] = 1  # Prevent division by zero
    pheromone = pheromone / pheromone_sum

    return heuristic, pheromone