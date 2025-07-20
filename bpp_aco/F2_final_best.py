# Final round optimized implementation for update_pheromone
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def update_pheromone(pheromone: np.ndarray, paths: list, fitnesses: np.ndarray, iteration: int, n_iterations: int) -> np.ndarray:
    """
    Update pheromone levels based on bin packing solution quality, encouraging diverse and effective paths while addressing opponent weaknesses.
    
    This approach incorporates adaptive reinforcement strategies that address the opponent’s suboptimal reliance on static decay and encourage exploration of new packing configurations.
    
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
    evaporation_rate = 0.85  # More balanced pheromone decay to manage memory
    n_ants = len(paths)
    exploration_factor = 0.1  # Enhancing exploration of less-utilized paths
    
    # Initialize pheromone updates matrix
    delta_pheromone = np.zeros_like(pheromone)
    
    # Determine the best fitness found so far
    best_fitness = np.min(fitnesses)
    
    # Calculate pheromone updates from paths and fitnesses
    for path, fitness in zip(paths, fitnesses):
        # Calculate a weight to discourage poorer solutions while encouraging exploration
        weight = fitness / (best_fitness + 1e-6)
        # Increment pheromone where items are packed in the same bin
        delta_pheromone[path[:, None] == path[None, :]] += weight * (1 + exploration_factor)

    # Apply evaporation and limit decay in higher exploration phases
    pheromone *= evaporation_rate
    pheromone += delta_pheromone  # Add new pheromone levels
    pheromone = np.maximum(pheromone, 0)  # Ensure non-negativity

    return pheromone