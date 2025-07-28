import numpy as np
import numba as nb
import random
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

@nb.njit(nb.void(nb.float32[:], nb.int32[:], nb.float32[:, :]), nogil=True, cache=usecache)
def calculate_population_costs(costs, population_flat, distances):
    """Vectorized cost calculation for entire population."""
    n_cities = distances.shape[0]
    pop_size = len(costs)
    
    for p in range(pop_size):
        cost = np.float32(0.0)
        start_idx = p * n_cities
        
        for i in range(n_cities):
            current_city = population_flat[start_idx + i]
            next_city = population_flat[start_idx + ((i + 1) % n_cities)]
            cost += distances[current_city, next_city]
        
        costs[p] = cost

@nb.njit(nb.int8(nb.int32[:], nb.int32[:]), nogil=True, cache=usecache)
def tours_equal(tour1, tour2):
    """Check if two tours represent the same cycle (considering rotations and reversals)."""
    n = len(tour1)
    if len(tour2) != n:
        return False
    
    # Find starting city (tour1[0]) in tour2
    start_positions = []
    for i in range(n):
        if tour2[i] == tour1[0]:
            start_positions.append(i)
    
    # Check each possible starting position
    for start_pos in start_positions:
        # Check forward direction
        match_forward = True
        for i in range(n):
            if tour1[i] != tour2[(start_pos + i) % n]:
                match_forward = False
                break
        if match_forward:
            return True
        
        # Check reverse direction
        match_reverse = True
        for i in range(n):
            if tour1[i] != tour2[(start_pos - i) % n]:
                match_reverse = False
                break
        if match_reverse:
            return True
    
    return False

@nb.njit(nb.int8(nb.int32[:], nb.int32), nogil=True, cache=usecache)
def is_valid_tour_numba(tour, n_cities):
    """Numba optimized tour validation."""
    if len(tour) != n_cities:
        return False
    
    visited = np.zeros(n_cities, dtype=nb.int8)
    for i in range(len(tour)):
        city = tour[i]
        if city < 0 or city >= n_cities or visited[city]:
            return False
        visited[city] = 1
    
    return True

def is_duplicate_tour_fast(tour_array, existing_tours_flat, n_cities, num_existing):
    """Fast duplicate checking using flattened arrays."""
    for i in range(num_existing):
        start_idx = i * n_cities
        existing_tour = existing_tours_flat[start_idx:start_idx + n_cities]
        if tours_equal(tour_array, existing_tour):
            return True
    return False

@nb.njit(nb.int32[:](nb.float32[:, :]), nogil=True, cache=usecache)
def greedy_initialization_numba(edge_scores):
    """Initialize tour using precomputed edge scores with greedy construction."""
    n_cities = edge_scores.shape[0]
    if n_cities <= 1:
        return np.arange(n_cities, dtype=nb.int32)
    
    tour = np.zeros(n_cities, dtype=nb.int32)
    tour[0] = 0
    visited = np.zeros(n_cities, dtype=nb.int8)
    visited[0] = 1
    
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
        
        tour[step] = best_city
        visited[best_city] = 1
    
    return tour

def precompute_edge_scores(distances):
    """Precompute all edge scores once using F1."""
    n_cities = distances.shape[0]
    edge_scores = np.full((n_cities, n_cities), -np.inf, dtype=np.float32)
    
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                edge_scores[i, j] = edge_score(i, j, distances)
    
    return edge_scores

def greedy_initialization_optimized(edge_scores_precomputed):
    """Initialize tour using precomputed edge scores."""
    tour_array = greedy_initialization_numba(edge_scores_precomputed)
    return tour_array

def deconstruct_tour_optimized(tour_array, distances, destruction_rate=0.3):
    """Optimized deconstruct tour using numpy arrays."""
    n = len(tour_array)
    
    # Calculate badness scores for all cities using F2
    badness_scores = np.zeros(n, dtype=np.float32)
    tour_list = tour_array.tolist()  # Convert once for F2 compatibility
    
    for i in range(n):
        badness_scores[i] = city_badness(i, tour_list, distances)
    
    # Get indices of worst cities
    num_to_remove = max(1, int(n * destruction_rate))
    num_to_remove = min(num_to_remove, n - 2)
    
    worst_indices = np.argpartition(badness_scores, -num_to_remove)[-num_to_remove:]
    
    # Create boolean mask for removal
    removal_mask = np.zeros(n, dtype=bool)
    removal_mask[worst_indices] = True
    
    # Split efficiently
    removed_cities = tour_array[removal_mask].copy()
    incomplete_tour = tour_array[~removal_mask].copy()
    
    return removed_cities, incomplete_tour

def repair_tour_optimized(removed_cities, incomplete_tour, distances):
    """Optimized repair tour using numpy arrays."""
    if len(removed_cities) == 0:
        return incomplete_tour.copy()
    
    current_tour = incomplete_tour.tolist()  # Convert for F3 compatibility
    
    # Insert each removed city at best position determined by F3
    for city in removed_cities:
        position = insert_position(int(city), current_tour, distances)
        current_tour.insert(position, int(city))
    
    return np.array(current_tour, dtype=np.int32)

def tournament_selection_batch(population_array, costs, tournament_size=3, num_selections=2):
    """Batch tournament selection for multiple parents."""
    pop_size = len(costs)
    n_cities = population_array.shape[1]
    
    selected_parents = np.zeros((num_selections, n_cities), dtype=np.int32)
    
    for sel in range(num_selections):
        # Random tournament indices
        tournament_indices = np.random.choice(pop_size, size=min(tournament_size, pop_size), replace=False)
        
        # Find best in tournament
        best_idx = tournament_indices[0]
        best_cost = costs[best_idx]
        
        for idx in tournament_indices[1:]:
            if costs[idx] < best_cost:
                best_cost = costs[idx]
                best_idx = idx
        
        selected_parents[sel] = population_array[best_idx]
    
    return selected_parents

def run_tsp_ga(distances: np.ndarray, 
               population_size: int = 50,
               generations: int = 100,
               destruction_rate: float = 0.3,
               elite_size: int = 5,
               seed: int = None) -> float:
    """
    Optimized Deconstruct and Repair Genetic Algorithm for TSP.
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    n_cities = distances.shape[0]
    distances_float32 = distances.astype(np.float32)
    
    # Precompute edge scores once
    edge_scores_precomputed = precompute_edge_scores(distances)
    
    # Initialize population as 2D numpy array for better memory efficiency
    population_array = np.zeros((population_size, n_cities), dtype=np.int32)
    valid_count = 0
    
    # Generate initial population
    for i in range(population_size):
        if valid_count < population_size // 2:  # First half using greedy
            tour = greedy_initialization_optimized(edge_scores_precomputed)
        else:  # Second half using random
            tour = np.arange(n_cities, dtype=np.int32)
            np.random.shuffle(tour)
        
        if is_valid_tour_numba(tour, n_cities):
            population_array[valid_count] = tour
            valid_count += 1
    
    # Fill remaining slots with random tours if needed
    while valid_count < population_size:
        tour = np.arange(n_cities, dtype=np.int32)
        np.random.shuffle(tour)
        if is_valid_tour_numba(tour, n_cities):
            population_array[valid_count] = tour
            valid_count += 1
    
    # Pre-allocate arrays for costs and elite storage
    costs = np.zeros(population_size, dtype=np.float32)
    elite_tours = np.zeros((elite_size, n_cities), dtype=np.int32)
    elite_costs = np.full(elite_size, np.inf, dtype=np.float32)
    
    best_cost = float('inf')
    
    for generation in range(generations):
        # Vectorized fitness calculation
        population_flat = population_array.flatten()
        calculate_population_costs(costs, population_flat, distances_float32)
        
        # Track best solution
        gen_best_idx = np.argmin(costs)
        gen_best_cost = float(costs[gen_best_idx])
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
        
        # Elite selection: keep best unique tours
        sorted_indices = np.argsort(costs)
        elite_count = 0
        
        for idx in sorted_indices:
            if elite_count >= elite_size:
                break
            
            candidate_tour = population_array[idx]
            candidate_cost = costs[idx]
            
            # Simple duplicate check for elites
            is_duplicate = False
            for e in range(elite_count):
                if tours_equal(candidate_tour, elite_tours[e]):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                elite_tours[elite_count] = candidate_tour
                elite_costs[elite_count] = candidate_cost
                elite_count += 1
        
        # Create new population
        new_population = np.zeros((population_size, n_cities), dtype=np.int32)
        new_count = 0
        
        # Add elites
        for e in range(elite_count):
            new_population[new_count] = elite_tours[e]
            new_count += 1
        
        # Generate offspring
        while new_count < population_size:
            # Batch parent selection
            parents = tournament_selection_batch(population_array, costs, tournament_size=3, num_selections=2)
            
            # Deconstruct both parents
            removed1, partial1 = deconstruct_tour_optimized(parents[0], distances, destruction_rate)
            removed2, partial2 = deconstruct_tour_optimized(parents[1], distances, destruction_rate)
            
            # Cross-repair
            offspring1 = repair_tour_optimized(removed1, partial1, distances)
            offspring2 = repair_tour_optimized(removed2, partial2, distances)
            
            # Add valid offspring
            if new_count < population_size and is_valid_tour_numba(offspring1, n_cities):
                new_population[new_count] = offspring1
                new_count += 1
            
            if new_count < population_size and is_valid_tour_numba(offspring2, n_cities):
                new_population[new_count] = offspring2
                new_count += 1
        
        # Update population
        population_array = new_population
    
    return best_cost