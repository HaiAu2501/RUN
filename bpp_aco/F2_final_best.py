# Final round optimized implementation for update_pheromone
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def update_pheromone(pheromone: np.ndarray, paths: list, fitnesses: np.ndarray, iteration: int, n_iterations: int) -> np.ndarray:
    """
    Update pheromone levels based on bin packing solution quality by optimizing exploration and boosting relevance of solutions.
    """
    # Hyperparameters
    evaporation_rate = 0.85  # Slightly higher than opponent, allows for better adaptation
    reinforcement_weight = 1.5  # Reduced weight to prevent over-reinforcement of suboptimal paths
    n_ants = len(paths)  
    
    # Initialize delta pheromone matrix
    delta_pheromone = np.zeros_like(pheromone)
    max_fitness = np.max(fitnesses) if fitnesses.size > 0 else 1  # Guard against division by zero

    # Reinforce pheromone levels from best performing ants with incremental learning
    for path, fitness in zip(paths, fitnesses):
        if fitness > 0:
            contribution = reinforcement_weight * (fitness / max_fitness)
            # New strategy: Increase contribution for paths not considered in previous iterations
            contribution += np.count_nonzero(pheromone[path[:, None] == path[None, :]]) * 0.1
            delta_pheromone[path[:, None] == path[None, :]] += contribution

    # Apply evaporation to main pheromone levels
    pheromone *= evaporation_rate

    # Integrate updates from ants and prevent zero division issues
    pheromone += delta_pheromone / (n_ants + 1e-5)
    pheromone = np.clip(pheromone, 0, None)  # Ensure non-negative pheromone levels

    return pheromone