import numpy as np
from typing import Tuple, List
from math import floor
from F1_final_best import initialize
from F2_final_best import update_pheromone

def is_valid_solution(best_path, demands, capacity, final_bins):
    if best_path is None:
        raise ValueError("No solution found")
    
    if len(best_path) != len(demands):
        raise ValueError(f"Path length {len(best_path)} != number of items {len(demands)}")
    
    # Check bin assignments are valid
    max_bin = np.max(best_path)
    if max_bin + 1 != final_bins:
        raise ValueError(f"Bin count mismatch: max_bin {max_bin + 1} != final_bins {final_bins}")
    
    # Check capacity constraints
    bin_loads = np.zeros(final_bins)
    for item_idx, bin_idx in enumerate(best_path):
        if bin_idx < 0 or bin_idx >= final_bins:
            raise ValueError(f"Invalid bin assignment: item {item_idx} -> bin {bin_idx}")
        bin_loads[bin_idx] += demands[item_idx]
    
    # Check no bin exceeds capacity
    for bin_idx, load in enumerate(bin_loads):
        if load > capacity:
            raise ValueError(f"Bin {bin_idx} exceeds capacity: {load} > {capacity}")
    
    # Check all items are assigned
    if len(best_path) != len(demands):
        raise ValueError("Not all items are assigned to bins")

def organize_path(path: np.ndarray) -> Tuple[int, np.ndarray]:
    """Organize path to get number of bins and reorganized assignments."""
    order = {}
    result = np.zeros_like(path)
    for i, v in enumerate(path):
        if v in order:
            result[i] = order[v]
        else:
            result[i] = order[v] = len(order)
    return len(order), result

def calculate_path_fitness(vacancies: List[int], capacity: int) -> float:
    """Calculate fitness based on bin utilization efficiency."""
    occupied = capacity - np.array(vacancies, dtype=float)
    result = ((occupied / capacity) ** 2).sum() / len(vacancies)
    return result

def uniform_number_generator(batch_size=500):
    """Generate random numbers in batches for efficiency."""
    while True:
        numbers = np.random.random(batch_size)
        for n in numbers:
            yield n

uniform_gen = uniform_number_generator()

def random_sample_discrete_distribution(prob: np.ndarray) -> int:
    """Stochastic sampling using discrete distribution."""
    cumprob = np.cumsum(prob)
    sampled = np.searchsorted(cumprob, next(uniform_gen) * cumprob[-1])
    return sampled if sampled < len(cumprob) else len(cumprob) - 1

def sample_path(demands: np.ndarray, capacity: int, prob: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Sample a single ant path for bin packing."""
    problem_size = len(demands)
    
    path = np.ones(problem_size, dtype=int) * -1
    valid_items = np.ones(problem_size, dtype=bool)
    current_bin = 0
    item_count = 0
    vacancies = []
    bin_vacancy = capacity
    bin_items = np.zeros_like(valid_items)
    ordinal = np.arange(problem_size, dtype=int)

    def random_select(mask: np.ndarray) -> int:
        valid = ordinal[mask]
        return valid[floor(next(uniform_gen) * len(valid))]

    for _ in range(problem_size):
        mask = np.bitwise_and(demands <= bin_vacancy, valid_items)
        
        if not np.any(mask):
            # No item fits, move to next bin
            vacancies.append(bin_vacancy)
            bin_vacancy = capacity
            item_count = 0
            current_bin += 1
            bin_items[:] = False
            selected = random_select(valid_items)
        else:
            if item_count == 0:
                # First item in bin
                selected = random_select(mask)
            else:
                # Select based on pheromone with items already in bin
                item_prob = (prob[bin_items].sum(0) / item_count + 1e-5) * mask
                selected = random_sample_discrete_distribution(item_prob)
        
        # Place item in current bin
        bin_items[selected] = True
        bin_vacancy -= demands[selected]
        valid_items[selected] = False
        path[selected] = current_bin
        item_count += 1
    
    vacancies.append(bin_vacancy)
    fitness = calculate_path_fitness(vacancies, capacity)
    
    return path, len(vacancies), fitness

def run_bpp_aco(demands: np.ndarray, capacity: int, n_ants: int = 20, n_iterations: int = 50, seed: int = 0) -> int:
    """Run Ant Colony Optimization for Bin Packing Problem."""
    np.random.seed(seed)
    
    problem_size = len(demands)
    alpha = 1.0  # pheromone influence
    beta = 1.0   # heuristic influence
    
    # Initialize using F1
    heuristic, pheromone = initialize(demands.copy(), capacity)

    # Ensure heuristic and pheromone are finite and non-negative
    heuristic = np.nan_to_num(heuristic, nan=0.01, posinf=1e10, neginf=0.01)
    pheromone = np.nan_to_num(pheromone, nan=0.01, posinf=1e10, neginf=0.01)
    
    # Ensure minimum positive values
    heuristic = np.maximum(heuristic, 0.01)
    pheromone = np.maximum(pheromone, 0.01)
    
    best_cost = problem_size
    best_path = np.arange(problem_size)

    list_objs = []
    
    # Main ACO loop
    for iteration in range(n_iterations + 1):
        # Calculate probability matrix
        prob = (pheromone ** alpha) * (heuristic ** beta)
        
        # Generate ant solutions
        paths, costs, fitnesses = [], [], []
        for _ in range(n_ants):
            path, cost, fitness = sample_path(demands, capacity, prob)
            paths.append(path)
            costs.append(cost)
            fitnesses.append(fitness)
        
        costs = np.array(costs)
        fitnesses = np.array(fitnesses)
        
        # Update best solution
        best_index = costs.argmin()
        if costs[best_index] < best_cost:
            best_cost = costs[best_index]
            best_path = paths[best_index]
        
        # Update pheromone using F2
        pheromone = update_pheromone(pheromone, paths, fitnesses, iteration, n_iterations)
        pheromone = np.nan_to_num(pheromone, nan=0.01, posinf=1.0, neginf=0.01)
        pheromone = np.maximum(pheromone, 0.01)  # Ensure minimum
        if iteration % 5 == 0:
        # Store objective values for analysis
            list_objs.append(best_cost)
    
    final_bins, _ = organize_path(best_path)

    is_valid_solution(best_path, demands, capacity, final_bins)
    
    return np.array(list_objs)

import os
import sys
import numpy as np

# Problem constants
N_ANTS = 20
N_ITERATIONS = 50

def eval_instance(instance_data, n_ants, n_iter, seed):
    """
    Evaluate a single BPP instance.
    
    Parameters
    ----------
    instance_data : np.ndarray
        Instance data: [capacity, demand1, demand2, ...]
    n_ants : int
        Number of ants.
    n_iter : int
        Number of iterations.
    seed : int
        Random seed.
        
    Returns
    -------
    int
        Number of bins used.
    """
    capacity = int(instance_data[0])
    demands = instance_data[1:].astype(int)
    
    return run_bpp_aco(demands, capacity, n_ants, n_iter, seed)

def process_file(path, n_ants, n_iter):
    """
    Process a dataset file.
    
    Parameters
    ----------
    path : str
        Path to dataset file.
    n_ants : int
        Number of ants.
    n_iter : int
        Number of iterations.
        
    Returns
    -------
    np.ndarray
        Results for all instances.
    """
    # Load dataset
    data = np.load(path)
    instances = data['instances']  # shape: (n_instances, n_items+1)
    
    n_instances = instances.shape[0]
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        result = eval_instance(instances[i], n_ants, n_iter, int(seeds[i]))
        results.append(result)
    
    return np.array(results)

def run(size):
    """
    Main evaluation function.
    
    Parameters
    ----------
    size : int
        Size of the BPP instance.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'datasets', f'test_BPP{size}.npz')

    if not os.path.exists(path):
        print(f"Warning: File {path} not found. Skipping.")
        return

    costs = process_file(path, n_ants=N_ANTS, n_iter=N_ITERATIONS)
    print(costs.mean(axis=0).tolist())

if __name__ == "__main__":
    # Get size from command line argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(size)