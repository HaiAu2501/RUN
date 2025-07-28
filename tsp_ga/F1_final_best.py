import numpy as np

def initialize_population(n_cities: int, population_size: int, distances: np.ndarray) -> np.ndarray:
    """
    Initialize population with real-valued encodings or permutation-based tours.
    
    This function can return either:
    1. Real-valued encodings in [0, 1] range 
    2. Permutation-based tours as city indices

    Don't need to care about output type here, I will handle it later.
    
    Parameters
    ----------
    n_cities : int
        Number of cities in TSP.
    population_size : int
        Size of population to create.
    distances : np.ndarray, shape (n_cities, n_cities)
        Distance matrix between cities, can be used for more complex initializations.

    Returns
    -------
    np.ndarray, shape (population_size, n_cities)
        Population matrix where each row represents an individual.
        Can be either:
        - Real-valued encodings
        - Integer permutations
    """
    # Simple uniform random initialization in [0, 1] range
    population = np.random.random((population_size, n_cities))
    
    return population