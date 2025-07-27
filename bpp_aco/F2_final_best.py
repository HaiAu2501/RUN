# Final round optimized implementation for update_pheromone
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def update_pheromone(pheromone: np.ndarray, paths: list, fitnesses: np.ndarray, iteration: int, n_iterations: int) -> np.ndarray:
    """
    Update pheromone levels based on bin packing solution quality with emphasis on diversity and performance.
    This strategy optimizes the pheromone update process via selective reinforcement and significant penalties for non-viable solutions.
    
    Parameters
    ----------
    pheromone : np.ndarray, shape (n, n)
        Current pheromone distribution matrix.
    paths : list[np.ndarray]
        List of ant solutions. Each solution is an array of shape (n,).
    fitnesses : np.ndarray, shape (n_ants,)
        Fitness values based on bin utilization efficiency.
    iteration : int
        Current iteration number.
    n_iterations : int
        Total number of iterations.

    Returns
    -------
    np.ndarray, shape (n, n)
        Updated pheromone levels after learning from packing solution quality.
    """
    # Hyperparameters
    decay = 0.85  # Slightly lower decay for longer pheromone lifespan
    successful_weight = 5.0  # Increased weight for successful solutions
    failure_penalty = 2.0  # Strong penalty for negative contribution
    n_ants = len(paths)

    # Initialize delta pheromone matrix
    delta_pheromone = np.zeros_like(pheromone)
    max_fitness = np.max(fitnesses) if np.any(fitnesses) else 1.0  # Prevent division by zero

    # Reinforce solutions based on fitness and apply strong penalties for poor solutions
    for path, fitness in zip(paths, fitnesses):
        if fitness > 0:
            # Strong contribution based on successful fitness
            contribution = (fitness / max_fitness) * successful_weight
            delta_pheromone[path[:, None] == path[None, :]] += contribution
        else:
            # Apply a penalty for each item in the path when fitness is non-positive
            unique_items = np.unique(path)
            for item in unique_items:
                delta_pheromone[item, item] -= failure_penalty / len(unique_items)  # Distributing penalty

    # Evaporation to balance the pheromone
    pheromone *= decay
    pheromone += delta_pheromone / (n_ants + 1e-10)  # Avoid division by zero

    # Ensure no negative pheromone levels
    pheromone = np.clip(pheromone, 0, None)
    return pheromone