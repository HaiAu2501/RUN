import numpy as np

def mutation(encoding: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Perform simple random mutation on encoding.
    
    Simple Gaussian noise mutation: add small random values to encoding.
    
    Parameters
    ----------
    encoding : np.ndarray, shape (n_cities,)
        Encoding to mutate (real values).
    distances : np.ndarray, shape (n_cities, n_cities)
        Distance matrix between cities, can be used for more complex mutation logic.
        
    Returns
    -------
    np.ndarray, shape (n_cities,)
        Mutated encoding.
    """
    mutated = encoding.copy()
    
    # Add Gaussian noise to each gene
    mutation_strength = 0.1  # Standard deviation of noise
    
    for i in range(len(mutated)):
        # Add Gaussian noise
        noise = np.random.normal(0.0, mutation_strength)
        mutated[i] += noise
    
    return mutated