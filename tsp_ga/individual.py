import numpy as np
import numba as nb

usecache = True

class Individual:
    """
    Individual representing a TSP solution with real-valued encoding.
    """
    
    def __init__(self, encoding: np.ndarray, distances: np.ndarray = None):
        """
        Initialize individual with real-valued encoding.
        
        Parameters
        ----------
        encoding : np.ndarray, shape (n_cities,)
            Real-valued vector representing the solution (should be pre-clipped).
        distances : np.ndarray, shape (n_cities, n_cities), optional
            Distance matrix for cost calculation.
        """
        self.encoding = encoding  # Assume pre-clipped by caller
        self.n_cities = len(encoding)
        self._tour = None
        self._cost = None
        
        if distances is not None:
            self.distances = distances
    
    @property
    def tour(self) -> np.ndarray:
        """Get tour representation (lazy evaluation)."""
        if self._tour is None:
            self._tour = encoding_to_tour(self.encoding)
        return self._tour
    
    @property
    def cost(self) -> float:
        """Get tour cost (lazy evaluation)."""
        if self._cost is None and hasattr(self, 'distances'):
            self._cost = calculate_tour_cost(self.tour, self.distances)
        return self._cost
    
    def invalidate_cache(self):
        """Invalidate cached tour and cost when encoding changes."""
        self._tour = None
        self._cost = None
    
    def set_encoding(self, new_encoding: np.ndarray):
        """Set new encoding and invalidate cache (assume pre-clipped)."""
        self.encoding = new_encoding  # Assume pre-clipped by caller
        self.invalidate_cache()

@nb.njit(nb.uint16[:](nb.float32[:]), nogil=True, cache=usecache)
def encoding_to_tour(encoding):
    """
    Convert real-valued encoding to tour using argsort.
    
    Parameters
    ----------
    encoding : np.ndarray, shape (n_cities,)
        Real-valued encoding.
        
    Returns
    -------
    np.ndarray, shape (n_cities,)
        Tour as permutation of city indices.
    """
    return np.argsort(encoding).astype(nb.uint16)

@nb.njit(nb.float32(nb.uint16[:], nb.float32[:, :]), nogil=True, cache=usecache)
def calculate_tour_cost(tour, distances):
    """
    Calculate total cost of a tour (numba optimized).
    
    Parameters
    ----------
    tour : np.ndarray, shape (n_cities,)
        Tour as array of city indices.
    distances : np.ndarray, shape (n_cities, n_cities)
        Distance matrix.
        
    Returns
    -------
    float
        Total tour cost.
    """
    n_cities = len(tour)
    cost = distances[tour[n_cities - 1], tour[0]]  # Return to start
    for i in range(n_cities - 1):
        cost += distances[tour[i], tour[i + 1]]
    return cost