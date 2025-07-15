import numpy as np

def initialize(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize heuristic and pheromone matrices for route optimization.
    This implementation blends robust heuristics with adaptive pheromone patterns for improved route selection.

    Parameters
    ----------
    distances : np.ndarray, shape (n_cities, n_cities)
        Matrix of pairwise distances between cities; entries must be positive.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        heuristic : np.ndarray, shape (n_cities, n_cities)
            Matrix representing desirability of traveling between cities based on an inverse scaling approach.
        pheromone : np.ndarray, shape (n_cities, n_cities)
            Matrix representing initial pheromone levels adjusted for exploration efficiency.
    """
    # Handle zero distances to avoid division issues
    distances = np.where(distances == 0, np.inf, distances)

    # Heuristic: encourage travel towards closer nodes, normalized and enhanced
    heuristic = 1.0 / distances
    heuristic = np.where(distances < np.mean(distances), heuristic, 0)  # Emphasize shorter routes

    # Pheromone initialization with uniform base and added variability
    pheromone_base = 0.15
    pheromone = np.full_like(distances, pheromone_base)
    pheromone += np.random.rand(*pheromone.shape) * 0.1  # Enhance pheromone variability
    pheromone[np.diag_indices_from(pheromone)] = 0  # Avoid self pheromone influence

    # Normalize pheromone values to ensure they guide exploration effectively
    mean_pheromone = pheromone.mean() + 1e-6  # Prevent zero division
    pheromone /= mean_pheromone  # Ensure range [0,1]

    return heuristic, pheromone
