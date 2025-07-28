import numpy as np

def crossover(parent1: np.ndarray, parent2: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform uniform crossover between two parent encodings.
    
    Simple uniform crossover: each gene comes from parent1 or parent2 with 50% probability.
    
    Parameters
    ----------
    parent1 : np.ndarray, shape (n_cities,)
        First parent encoding (real values).
    parent2 : np.ndarray, shape (n_cities,)
        Second parent encoding (real values).
    distances : np.ndarray, shape (n_cities, n_cities)
        Distance matrix between cities, can be used for more complex crossover logic.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        offspring1, offspring2 : Two offspring encodings.
    """
    n_cities = len(parent1)
    
    offspring1 = np.zeros(n_cities)
    offspring2 = np.zeros(n_cities)
    
    # Uniform crossover: each gene comes from random parent
    for i in range(n_cities):
        if np.random.random() < 0.5:
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]
        else:
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]
    
    return offspring1, offspring2