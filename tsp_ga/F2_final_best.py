import numpy as np

def city_badness(tour_idx: int, tour: list[int], distances: np.ndarray) -> float:
    """
    Calculate badness score of city at tour_idx in the tour.
    Higher score = worse city (should be removed first).
    
    Parameters
    ----------
    tour_idx : int
        Index in tour (not city ID)
    tour : list[int]
        Current tour as list of city IDs
    distances : np.ndarray, shape (n, n)
        Distance matrix between cities
        
    Returns
    -------
    float
        Badness score. Higher = worse city.
    """
    n = len(tour)
    
    city = tour[tour_idx]
    prev_city = tour[(tour_idx - 1) % n]
    next_city = tour[(tour_idx + 1) % n]
    
    # Badness = sum of distances to neighbors
    # Cities with long connections are "bad"
    return distances[prev_city, city] + distances[city, next_city]