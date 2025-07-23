# Final round optimized implementation for update_pheromone
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def update_pheromone(pheromone: np.ndarray, paths: list, fitnesses: np.ndarray, iteration: int, n_iterations: int) -> np.ndarray:
    """
    Update pheromone levels based on bin packing solution quality.
    
    This strategy enhances exploration by further balancing pheromone reinforcement between unique successful paths and diverse solutions. It uses a reduced decay for memory persistence and dynamic scaling based on the average fitness of solutions.
    
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
    evaporation_rate = 0.85  # Lower decay to keep more valuable pheromone
    contribution_scale = 3.0  # Increased importance on well-performing solutions
    exploration_boost = 0.1   # Boost for minor paths to maintain exploration
    n_ants = len(paths)  # Number of ant solutions

    # Initialize delta pheromone contributions matrix
    delta_pheromone = np.zeros_like(pheromone)
    max_fitness = np.max(fitnesses) if np.max(fitnesses) > 0 else 1  # Guard against division by zero

    # Use a dictionary to accumulate unique path contributions
    unique_path_contributions = {}  
    for path, fitness in zip(paths, fitnesses):
        if fitness > 0:
            path_tuple = tuple(path)
            if path_tuple not in unique_path_contributions:
                # Compute contribution for the path's average fitness 
                avg_fitness = np.mean(fitnesses)  # Use overall average to scale rewards
                contribution = avg_fitness * contribution_scale / max_fitness
                unique_path_contributions[path_tuple] = contribution

    # Apply contributions to delta pheromone based on unique paths
    for path_tuple, contribution in unique_path_contributions.items():
        indices = np.array(path_tuple)[:, None] == np.array(path_tuple)[None, :]
        delta_pheromone[indices] += contribution + exploration_boost  # Include exploration boost

    # Apply pheromone evaporation
    pheromone *= evaporation_rate
    # Reinstate contributions based on unique contributions
    if len(unique_path_contributions) > 0:
        pheromone += delta_pheromone / len(unique_path_contributions)  # Normalize by the number of unique contributions 
    pheromone = np.clip(pheromone, 0, None)  # Ensure non-negative values
    
    return pheromone
