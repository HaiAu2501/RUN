import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from F1_final_best import edge_score
from F2_final_best import city_badness  
from F3_final_best import insert_position

usecache = True

@nb.njit(nb.float32(nb.int32[:], nb.float32[:, :]), nogil=True, cache=usecache)
def calculate_tour_cost(tour, distances):
    """Calculate total cost of a tour."""
    n = len(tour)
    if n < 2:
        return np.float32(np.inf)
    
    cost = np.float32(0.0)
    for i in range(n):
        cost += distances[tour[i], tour[(i + 1) % n]]
    return cost

@nb.njit(nb.int8(nb.int32[:], nb.int32), nogil=True, cache=usecache)
def is_valid_tour(tour, n_cities):
    """Check if tour is valid (visits all cities exactly once)."""
    if len(tour) != n_cities:
        return False
    
    visited = np.zeros(n_cities, dtype=nb.int8)
    for i in range(len(tour)):
        city = tour[i]
        if city < 0 or city >= n_cities or visited[city]:
            return False
        visited[city] = 1
    
    return True

@nb.njit(nb.int32[:](nb.float32[:, :], nb.int32), nogil=True, cache=usecache)
def greedy_construction(edge_scores, start_city):
    """Construct tour greedily using F1 edge scores from given start city."""
    n_cities = edge_scores.shape[0]
    if n_cities <= 1:
        return np.arange(n_cities, dtype=nb.int32)
    
    tour = np.zeros(n_cities, dtype=nb.int32)
    tour[0] = start_city
    visited = np.zeros(n_cities, dtype=nb.int8)
    visited[start_city] = 1
    
    # Greedy construction using F1 scores
    for step in range(1, n_cities):
        current_city = tour[step - 1]
        
        best_city = -1
        best_score = np.float32(-np.inf)
        
        for next_city in range(n_cities):
            if visited[next_city] == 0:
                score = edge_scores[current_city, next_city]
                if score > best_score:
                    best_score = score
                    best_city = next_city
        
        if best_city == -1:  # Fallback if no valid city found
            for next_city in range(n_cities):
                if visited[next_city] == 0:
                    best_city = next_city
                    break
        
        tour[step] = best_city
        visited[best_city] = 1
    
    return tour

@nb.njit(nb.void(nb.int32[:], nb.float32[:, :], nb.int32), nogil=True, cache=usecache)
def two_opt_numba(tour, distances, max_iterations=50):
    """Safe 2-opt local improvement with iteration limit."""
    n = len(tour)
    iteration = 0
    
    while iteration < max_iterations:
        improved = False
        best_improvement = 0.0
        best_i = -1
        best_j = -1
        
        # Find best improvement in this iteration
        for i in range(1, n - 1):
            for j in range(i + 2, n):  # Ensure j - i >= 2 to avoid adjacent edges
                if j >= n:
                    break
                
                # Calculate change in cost if we reverse segment [i:j]
                # Current edges: (i-1,i), (j,j+1)  
                # New edges: (i-1,j), (i,j+1)
                old_cost = (distances[tour[i-1], tour[i]] + 
                           distances[tour[j], tour[(j+1) % n]])
                new_cost = (distances[tour[i-1], tour[j]] + 
                           distances[tour[i], tour[(j+1) % n]])
                
                improvement = old_cost - new_cost
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i = i
                    best_j = j
        
        # Apply best improvement found
        if best_improvement > 1e-6:  # Small epsilon to avoid floating point issues
            # Reverse segment [best_i:best_j]
            left = best_i
            right = best_j
            while left < right:
                tour[left], tour[right] = tour[right], tour[left]
                left += 1
                right -= 1
            improved = True
        
        if not improved:
            break
            
        iteration += 1

def precompute_edge_scores(distances):
    """Precompute all edge scores using F1."""
    n_cities = distances.shape[0]
    edge_scores = np.full((n_cities, n_cities), -np.inf, dtype=np.float32)
    
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                edge_scores[i, j] = edge_score(i, j, distances)
    
    return edge_scores

def deconstruct_tour(tour_array, distances, destruction_rate=0.3):
    """
    Deconstruct tour by removing bad cities using F2.
    
    Args:
        tour_array: Current tour as numpy array
        distances: Distance matrix
        destruction_rate: Fraction of cities to remove
    
    Returns:
        removed_cities: Cities that were removed
        partial_tour: Remaining tour after removal
    """
    n = len(tour_array)
    
    # Calculate badness scores using F2
    badness_scores = np.zeros(n, dtype=np.float32)
    tour_list = tour_array.tolist()
    
    for i in range(n):
        badness_scores[i] = city_badness(i, tour_list, distances)
    
    # Determine cities to remove
    num_to_remove = max(1, int(n * destruction_rate))
    num_to_remove = min(num_to_remove, n - 3)  # Keep at least 3 cities
    
    # Get indices of worst cities
    worst_indices = np.argpartition(badness_scores, -num_to_remove)[-num_to_remove:]
    
    # Create removal mask
    removal_mask = np.zeros(n, dtype=bool)
    removal_mask[worst_indices] = True
    
    # Split tour
    removed_cities = tour_array[removal_mask].copy()
    partial_tour = tour_array[~removal_mask].copy()
    
    return removed_cities, partial_tour

def repair_tour(removed_cities, partial_tour, distances):
    """
    Repair tour by inserting removed cities using F3.
    
    Args:
        removed_cities: Cities to be inserted
        partial_tour: Current partial tour
        distances: Distance matrix
    
    Returns:
        Complete repaired tour
    """
    if len(removed_cities) == 0:
        return partial_tour.copy()
    
    current_tour = partial_tour.tolist()
    
    # Sort removed cities by distance to current tour (nearest first)
    if len(current_tour) > 0:
        city_distances = []
        for city in removed_cities:
            min_dist = min(distances[city, tour_city] for tour_city in current_tour)
            city_distances.append((min_dist, city))
        
        city_distances.sort()  # Sort by distance (nearest first)
        sorted_removed = [city for _, city in city_distances]
    else:
        sorted_removed = removed_cities.tolist()
    
    # Insert each removed city at best position using F3
    for city in sorted_removed:
        if len(current_tour) == 0:
            current_tour.append(int(city))
        else:
            position = insert_position(int(city), current_tour, distances)
            current_tour.insert(position, int(city))
    
    return np.array(current_tour, dtype=np.int32)

def single_dr_run(distances, start_city, destruction_rate=0.3, use_2opt=True):
    """
    Single Deconstruction-Repair run from given start city.
    
    Args:
        distances: Distance matrix
        start_city: Starting city for greedy construction
        destruction_rate: Fraction of cities to remove in deconstruction
        use_2opt: Whether to apply 2-opt improvement
    
    Returns:
        tour: Best tour found
        cost: Cost of the tour
    """
    try:
        distances_float32 = distances.astype(np.float32)
        n_cities = distances.shape[0]
        
        # 1. Precompute edge scores
        edge_scores = precompute_edge_scores(distances)
        edge_scores_float32 = edge_scores.astype(np.float32)
        
        # 2. Greedy construction using F1
        tour = greedy_construction(edge_scores_float32, start_city)
        
        # Validate initial tour
        if not is_valid_tour(tour, n_cities):
            # Fallback: simple sequential tour
            tour = np.arange(n_cities, dtype=np.int32)
            np.random.shuffle(tour[1:])  # Keep start_city at position 0
            tour[0] = start_city
        
        # 3. Deconstruction using F2
        removed_cities, partial_tour = deconstruct_tour(tour, distances, destruction_rate)
        
        # 4. Repair using F3
        repaired_tour = repair_tour(removed_cities, partial_tour, distances)
        
        # Validate repaired tour
        if not is_valid_tour(repaired_tour, n_cities):
            # Use original tour if repair failed
            repaired_tour = tour
        
        # 5. Optional 2-opt improvement
        if use_2opt and len(repaired_tour) > 3:
            two_opt_numba(repaired_tour, distances_float32, max_iterations=10)
        
        # Calculate final cost
        final_cost = float(calculate_tour_cost(repaired_tour, distances_float32))
        
        return repaired_tour, final_cost
        
    except Exception as e:
        # Fallback: return simple tour
        fallback_tour = np.arange(n_cities, dtype=np.int32)
        fallback_cost = float(calculate_tour_cost(fallback_tour, distances.astype(np.float32)))
        return fallback_tour, fallback_cost

def run_tsp_dr(distances: np.ndarray, 
               destruction_rate: float = 0.3,
               use_2opt: bool = True,
               max_workers: int = None,
               seed: int = None) -> float:
    """
    Deconstruction-Repair algorithm for TSP with parallel execution.
    
    Args:
        distances: Distance matrix (n x n)
        destruction_rate: Fraction of cities to remove (0.1 to 0.5)
        use_2opt: Whether to apply 2-opt local improvement
        max_workers: Maximum number of threads (None = n_cities)
        seed: Random seed for reproducibility
    
    Returns:
        Best tour cost found
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    n_cities = distances.shape[0]
    
    # Handle edge cases
    if n_cities <= 1:
        return 0.0
    if n_cities == 2:
        return float(2 * distances[0, 1])
    
    # Determine number of workers
    if max_workers is None:
        max_workers = n_cities
    else:
        max_workers = min(max_workers, n_cities)
    
    best_tour = None
    best_cost = float('inf')
    
    # Parallel execution: try all starting cities
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all starting cities
        futures = [
            executor.submit(single_dr_run, distances, start_city, 
                          destruction_rate, use_2opt)
            for start_city in range(n_cities)
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                tour, cost = future.result()
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour
            except Exception as e:
                # Skip failed runs
                continue
    
    # Validate final result
    if best_tour is None or best_cost == float('inf'):
        # Ultimate fallback
        fallback_tour = np.arange(n_cities, dtype=np.int32)
        best_cost = float(calculate_tour_cost(fallback_tour, distances.astype(np.float32)))
    
    return best_cost