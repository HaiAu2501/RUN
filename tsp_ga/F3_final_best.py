import numpy as np

def insert_position(city: int, incomplete_tour: list[int], distances: np.ndarray) -> int:
    """
    Find best position to insert city in incomplete_tour.
    
    Parameters
    ----------
    city : int
        City ID to insert
    incomplete_tour : list[int]
        Current incomplete tour
    distances : np.ndarray, shape (n, n)
        Distance matrix between cities
        
    Returns
    -------
    int
        Best position index to insert city (0 to len(incomplete_tour))
    """
    # Find nearest city in incomplete tour
    nearest_city_idx = 0
    min_dist = float('inf')
    
    for i, tour_city in enumerate(incomplete_tour):
        dist = distances[city, tour_city]
        if dist < min_dist:
            min_dist = dist
            nearest_city_idx = i
    
    # Insert next to nearest city (after it)
    return nearest_city_idx + 1