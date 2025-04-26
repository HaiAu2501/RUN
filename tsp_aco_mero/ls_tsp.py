import numpy as np
import numpy.typing as npt
import numba as nb
from numba import njit, prange
import sys

# Type aliases
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]

# Configuration
usecache = True


@njit(nb.float64[:,:](nb.float64[:,:], nb.int32), cache=usecache, nogil=True, fastmath=True)
def compute_heuristic(distances: FloatArray, iteration: int) -> FloatArray:
    """HeuristicImpl.compute() optimized with numba and type signatures"""
    n = distances.shape[0]
    
    # Avoid division by zero
    distances_safe = distances.copy()
    for i in range(n):
        for j in range(n):
            if distances_safe[i, j] == 0:
                distances_safe[i, j] = 1e-10
    
    # Dynamic factors
    dynamic_socio_economic_factor = np.abs(np.sin(iteration / 10.0) + 1.0) * 0.5 + 0.5
    dynamic_environmental_impact = np.random.rand(n, n) * 0.25
    
    # Enhanced heuristic calculation
    attractiveness = np.power((1.0 / distances_safe) * dynamic_socio_economic_factor, 2.0) * (1.0 - dynamic_environmental_impact)
    
    # Normalize
    for i in range(n):
        row_sum = np.sum(attractiveness[i, :])
        if row_sum > 0:
            attractiveness[i, :] /= row_sum
    
    return attractiveness


@njit((nb.float64[:,:], nb.float64[:,:], nb.int32, nb.float64[:], nb.float64[:]), 
      cache=usecache, nogil=True, fastmath=True)
def compute_probabilities(pheromone: FloatArray, heuristic: FloatArray, iteration: int, 
                         alpha_history: FloatArray, beta_history: FloatArray) -> tuple:
    """ProbabilityImpl.compute() optimized with numba and type signatures"""
    n = pheromone.shape[0]
    
    # Dynamic hyperparameters
    alpha = 1.0
    beta = 1.0
    
    if iteration > 0 and len(alpha_history) > 0:
        alpha_std = np.std(alpha_history)
        beta_std = np.std(beta_history)
        alpha += alpha_std * np.random.randn()
        beta += beta_std * np.random.randn()
    
    # Ensure positive values
    alpha = max(0.1, alpha)
    beta = max(0.1, beta)
    
    # Safe normalization
    pheromone_normalized = np.zeros((n, n), dtype=np.float64)
    heuristic_normalized = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        ph_sum = np.sum(pheromone[i, :])
        h_sum = np.sum(heuristic[i, :])
        
        if ph_sum > 0:
            pheromone_normalized[i, :] = pheromone[i, :] / ph_sum
        if h_sum > 0:
            heuristic_normalized[i, :] = heuristic[i, :] / h_sum
    
    # Calculate probabilities
    probabilities = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            probabilities[i, j] = (pheromone_normalized[i, j] ** alpha) * (heuristic_normalized[i, j] ** beta)
    
    # Normalize
    for i in range(n):
        prob_sum = np.sum(probabilities[i, :])
        if prob_sum > 0:
            probabilities[i, :] /= prob_sum
    
    return probabilities, alpha, beta


@njit((nb.float64[:,:], nb.int32[:,:], nb.float64[:], nb.int32), 
      cache=usecache, nogil=True, fastmath=True)
def update_pheromones(pheromone: FloatArray, paths: IntArray, costs: FloatArray, 
                     iteration: int) -> tuple:
    """PheromoneImpl.update() optimized with numba and type signatures"""
    n = pheromone.shape[0]
    n_ants = paths.shape[1]
    
    # Dynamic decay
    decay = 0.8 + (0.1 / (1.0 + iteration))
    competition_weight = 1.0
    
    # Evaporate
    current_pheromone = pheromone * decay
    
    # Prepare updates
    pheromone_updates = np.zeros((n, n), dtype=np.float64)
    
    # Rank ants
    ranked_indices = np.argsort(costs)
    
    # Update based on ranking
    for rank in range(n_ants):
        i = ranked_indices[rank]
        path = paths[:, i]
        cost = costs[i]
        
        contribution = competition_weight * (1.0 / cost) * (1.0 / (rank + 1.0))
        
        # Update path
        for j in range(n):
            current_city = path[j]
            next_city = path[(j + 1) % n]
            
            pheromone_updates[current_city, next_city] += contribution
            pheromone_updates[next_city, current_city] += contribution
    
    return current_pheromone + pheromone_updates, decay


@njit((nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.int32, nb.int32, nb.int32), 
      cache=usecache, nogil=True, fastmath=True)
def construct_solutions(pheromone: FloatArray, heuristic: FloatArray, probabilities: FloatArray, 
                       n_ants: int, n_cities: int, seed: int) -> IntArray:
    """Construct solutions according to original logic with type signatures"""
    np.random.seed(seed)
    
    # Choose random starting cities
    start = np.random.randint(0, n_cities, size=n_ants)
    
    # Initialize mask and paths
    mask = np.ones((n_ants, n_cities), dtype=np.float64)
    for i in range(n_ants):
        mask[i, start[i]] = 0.0
    
    paths = np.zeros((n_cities, n_ants), dtype=np.int32)
    paths[0, :] = start
    current_cities = start.copy()
    
    # Construct paths
    for step in range(1, n_cities):
        for ant in range(n_ants):
            current = current_cities[ant]
            probs = probabilities[current, :] * mask[ant, :]
            
            # Normalize
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs /= prob_sum
            else:
                # Fallback to unvisited cities
                unvisited = np.where(mask[ant, :] > 0)[0]
                if len(unvisited) > 0:
                    next_city = unvisited[np.random.randint(0, len(unvisited))]
                else:
                    next_city = np.random.randint(0, n_cities)
                paths[step, ant] = next_city
                current_cities[ant] = next_city
                mask[ant, next_city] = 0.0
                continue
            
            # Select next city
            cumsum = 0.0
            r = np.random.random()
            next_city = -1
            
            for j in range(n_cities):
                cumsum += probs[j]
                if cumsum >= r:
                    next_city = j
                    break
            
            if next_city == -1:
                next_city = np.argmax(probs)
            
            paths[step, ant] = next_city
            current_cities[ant] = next_city
            mask[ant, next_city] = 0.0
    
    return paths


@njit((nb.int32[:,:], nb.float64[:,:]), cache=usecache, nogil=True, parallel=True, fastmath=True)
def calculate_path_costs(paths: IntArray, distances: FloatArray) -> FloatArray:
    """Calculate costs with original logic and parallel processing"""
    n_ants = paths.shape[1]
    n_cities = paths.shape[0]
    costs = np.zeros(n_ants, dtype=np.float64)
    
    for ant in prange(n_ants):
        cost = 0.0
        for i in range(n_cities):
            current = paths[i, ant]
            next_city = paths[(i + 1) % n_cities, ant]
            cost += distances[current, next_city]
        costs[ant] = cost
    
    return costs


def solve_tsp_aco(distances: FloatArray, n_ants: int = 50, n_iterations: int = 100, 
                  seed: int = 123) -> float:
    """
    Solve TSP using ACO with original logic + numba optimization with type signatures
    """
    n_cities = distances.shape[0]
    
    # Initialize
    pheromone = np.ones_like(distances, dtype=np.float64)
    alpha_history = np.zeros(n_iterations, dtype=np.float64)
    beta_history = np.zeros(n_iterations, dtype=np.float64)
    decay_history = np.zeros(n_iterations, dtype=np.float64)
    
    best_cost = float('inf')
    
    # Main loop
    for iteration in range(n_iterations):
        # Compute heuristic
        heuristic = compute_heuristic(distances, iteration)
        
        # Compute probabilities
        probabilities, alpha, beta = compute_probabilities(
            pheromone, heuristic, iteration, alpha_history[:iteration], beta_history[:iteration]
        )
        
        # Store hyperparameters
        alpha_history[iteration] = alpha
        beta_history[iteration] = beta
        
        # Construct solutions
        paths = construct_solutions(
            pheromone, heuristic, probabilities, n_ants, n_cities, seed + iteration
        )
        
        # Calculate costs
        costs = calculate_path_costs(paths, distances)
        
        # Update best
        min_idx = np.argmin(costs)
        if costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
        
        # Update pheromones
        pheromone, decay = update_pheromones(pheromone, paths, costs, iteration)
        decay_history[iteration] = decay
    
    return best_cost

def run_aco(size):
    avg_costs = 0
    path = f"tsp_aco_mero/ls_tsp/TSP{size}.npy"
    prob_batch = np.load(path)
    from scipy.spatial import distance_matrix
    # Calculate the distance matrix
    for i, prob in enumerate(prob_batch):
        distances = distance_matrix(prob, prob)
        best_cost = solve_tsp_aco(distances, n_ants=100, n_iterations=200, seed=0)
        print(f"Cost for TSP{size} {i}: {best_cost}")
        avg_costs += best_cost
    avg_costs /= len(prob_batch)
    print(f"Average cost for TSP{size}: {avg_costs}")

size = sys.argv[1]

# Example usage
if __name__ == "__main__":
    run_aco(size)