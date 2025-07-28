import numpy as np
import numba as nb
from typing import List
from F1_final_best import initialize_population
from F2_final_best import crossover
from F3_final_best import mutation
from individual import Individual, calculate_tour_cost, encoding_to_tour

usecache = True

@nb.njit(nb.float32[:](nb.float32[:], nb.float32), nogil=True, cache=usecache)
def postprocess_encoding(encoding, noise_scale):
    """
    Post-process encoding to ensure proper [0, 1] range and avoid duplicate values.
    
    Steps:
    1. Add small random noise to avoid duplicate values
    2. Min-max scale to [0, 1] range
    
    Parameters
    ----------
    encoding : np.ndarray, shape (n_cities,)
        Raw encoding that may have any values or duplicates.
    noise_scale : float
        Standard deviation of Gaussian noise to add.
        
    Returns
    -------
    np.ndarray, shape (n_cities,)
        Post-processed encoding in [0, 1] range with unique values.
    """
    n_cities = len(encoding)
    processed = np.zeros(n_cities, dtype=nb.float32)
    
    # Step 1: Add small random noise to avoid duplicates
    for i in range(n_cities):
        noise = np.random.normal(0.0, noise_scale)
        processed[i] = encoding[i] + noise
    
    # Step 2: Min-max scale to [0, 1] range
    min_val = processed[0]
    max_val = processed[0]
    
    # Find min and max
    for i in range(1, n_cities):
        if processed[i] < min_val:
            min_val = processed[i]
        if processed[i] > max_val:
            max_val = processed[i]
    
    # Scale to [0, 1] range
    range_val = max_val - min_val
    if range_val > 1e-12:  # Avoid division by zero
        for i in range(n_cities):
            processed[i] = (processed[i] - min_val) / range_val
    else:
        # If all values are the same, create uniform distribution
        for i in range(n_cities):
            processed[i] = float(i) / float(n_cities - 1) if n_cities > 1 else 0.0
    
    return processed

@nb.njit(nb.float32[:](nb.float32[:]), nogil=True, cache=usecache)
def clip_encoding(encoding):
    """
    Clip single encoding to [0, 1] range (numba optimized).
    Only needed after F2/F3 operations that might go out of bounds.
    
    Parameters
    ----------
    encoding : np.ndarray, shape (n_cities,)
        Encoding to clip.
        
    Returns
    -------
    np.ndarray, shape (n_cities,)
        Clipped encoding in [0, 1] range.
    """
    n_cities = len(encoding)
    clipped = np.zeros(n_cities, dtype=nb.float32)
    
    for i in range(n_cities):
        val = encoding[i]
        if val < 0.0:
            clipped[i] = 0.0
        elif val > 1.0:
            clipped[i] = 1.0
        else:
            clipped[i] = val
    
    return clipped

@nb.njit(nb.float32[:](nb.float32[:,:], nb.float32[:,:]), nogil=True, cache=usecache)
def calculate_population_costs(population_encodings, distances):
    """
    Calculate costs for entire population (numba optimized).
    
    Parameters
    ----------
    population_encodings : np.ndarray, shape (population_size, n_cities)
        Population of real-valued encodings.
    distances : np.ndarray, shape (n_cities, n_cities)
        Distance matrix.
        
    Returns
    -------
    np.ndarray, shape (population_size,)
        Cost values for each individual.
    """
    population_size = population_encodings.shape[0]
    costs = np.zeros(population_size, dtype=nb.float32)
    
    for i in range(population_size):
        # Convert encoding to tour
        tour = encoding_to_tour(population_encodings[i])
        # Calculate cost
        costs[i] = calculate_tour_cost(tour, distances)
    
    return costs

@nb.njit(nb.uint16(nb.float32[:], nb.uint16), nogil=True, cache=usecache)
def tournament_selection(costs, tournament_size):
    """
    Tournament selection returning index of winner (numba optimized).
    
    Parameters
    ----------
    costs : np.ndarray, shape (population_size,)
        Cost values (lower is better).
    tournament_size : int
        Number of individuals in tournament.
        
    Returns
    -------
    int
        Index of selected individual.
    """
    population_size = len(costs)
    
    # Select random individuals for tournament
    best_idx = np.random.randint(0, population_size)
    best_cost = costs[best_idx]
    
    for _ in range(tournament_size - 1):
        candidate_idx = np.random.randint(0, population_size)
        candidate_cost = costs[candidate_idx]
        
        if candidate_cost < best_cost:
            best_idx = candidate_idx
            best_cost = candidate_cost
    
    return best_idx

def run_tsp_ga(distances: np.ndarray, population_size: int = 100, 
               generations: int = 500, crossover_rate: float = 0.8,
               mutation_rate: float = 0.1, elite_size: int = 10,
               tournament_size: int = 3) -> float:
    """
    Run Genetic Algorithm for TSP with real-valued encoding.
    
    Parameters
    ----------
    distances : np.ndarray, shape (n_cities, n_cities)
        Distance matrix between cities.
    population_size : int
        Size of the population.
    generations : int
        Number of generations to evolve.
    crossover_rate : float
        Crossover probability.
    mutation_rate : float
        Mutation probability.
    elite_size : int
        Number of elite individuals to preserve.
    tournament_size : int
        Tournament selection size.
        
    Returns
    -------
    float
        Best tour cost found.
    """
    n_cities = distances.shape[0]
    
    # Convert distances to float32 for numba compatibility
    distances_f32 = distances.astype(np.float32)
    
    # Initialize population using F1
    population_encodings = initialize_population(n_cities, population_size, distances.copy())
    
    # Post-process initial population using shared function
    noise_scale = np.float32(1e-6)
    for i in range(population_size):
        population_encodings[i] = postprocess_encoding(population_encodings[i].astype(np.float32), noise_scale)
    
    # Convert to float32 for numba compatibility
    population_encodings = population_encodings.astype(np.float32)
    best_cost = float('inf')
    
    for generation in range(generations):
        # Calculate costs using numba
        costs = calculate_population_costs(population_encodings, distances_f32)
        
        # Track best solution
        best_idx = np.argmin(costs)
        current_cost = float(costs[best_idx])
        
        if current_cost < best_cost:
            best_cost = current_cost
        
        # Create new population array
        new_population = np.zeros_like(population_encodings)
        
        # Elitism: preserve best individuals
        elite_indices = np.argsort(costs)[:elite_size]
        for i, elite_idx in enumerate(elite_indices):
            new_population[i] = population_encodings[elite_idx].copy()
        
        # Generate offspring for remaining positions
        offspring_count = elite_size
        
        while offspring_count < population_size:
            # Selection using tournament
            parent1_idx = tournament_selection(costs, tournament_size)
            parent2_idx = tournament_selection(costs, tournament_size)
            
            parent1_encoding = population_encodings[parent1_idx]
            parent2_encoding = population_encodings[parent2_idx]
            
            # Crossover (apply rate logic and post-process output)
            if np.random.random() < crossover_rate:
                offspring1_enc, offspring2_enc = crossover(parent1_encoding, parent2_encoding, distances.copy())
                # Post-process crossover outputs to ensure proper encoding
                offspring1_enc = postprocess_encoding(offspring1_enc.astype(np.float32), noise_scale)
                offspring2_enc = postprocess_encoding(offspring2_enc.astype(np.float32), noise_scale)
            else:
                offspring1_enc = parent1_encoding.copy()
                offspring2_enc = parent2_encoding.copy()
            
            # Mutation (apply rate logic and post-process output)
            if np.random.random() < mutation_rate:
                offspring1_enc = mutation(offspring1_enc, distances.copy())
                offspring1_enc = postprocess_encoding(offspring1_enc.astype(np.float32), noise_scale)
            if np.random.random() < mutation_rate:
                offspring2_enc = mutation(offspring2_enc, distances.copy())
                offspring2_enc = postprocess_encoding(offspring2_enc.astype(np.float32), noise_scale)
            
            # Add to new population
            if offspring_count < population_size:
                new_population[offspring_count] = offspring1_enc
                offspring_count += 1
            
            if offspring_count < population_size:
                new_population[offspring_count] = offspring2_enc
                offspring_count += 1
        
        population_encodings = new_population
    
    return best_cost

def create_individual_from_encoding(encoding: np.ndarray, distances: np.ndarray) -> Individual:
    """
    Create Individual object from encoding.
    
    Parameters
    ----------
    encoding : np.ndarray
        Real-valued encoding.
    distances : np.ndarray
        Distance matrix.
        
    Returns
    -------
    Individual
        Individual object with encoding and distances.
    """
    return Individual(encoding, distances)