import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from concurrent.futures import ThreadPoolExecutor
from gls import run_tsp_gls

OPTIMAL = {
    "50": 5.68457994395107,
    "100": 7.778580370400294,
    "200": 10.71194600194464,
    "500": 16.499886342078646
}

def solve_instance(coordinates: np.ndarray, perturbation_moves: int, iter_limit: int, seed: int) -> float:
    """
    Solve a single TSP instance using GLS.
    
    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 2)
        City coordinates
    perturbation_moves : int
        Number of perturbation moves per GLS iteration
    iter_limit : int
        Maximum number of GLS iterations
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    float
        Best tour cost found
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create distance matrix from coordinates
    dist_matrix = distance_matrix(coordinates, coordinates)
    
    # Solve using GLS
    best_cost = run_tsp_gls(
        distmat=dist_matrix,
        perturbation_moves=perturbation_moves,
        iter_limit=iter_limit
    )
    
    # Calculate and return tour cost
    return best_cost

def process_file(path: str, perturbation_moves: int, iter_limit: int) -> np.ndarray:
    """
    Process a dataset file and solve all instances.
    
    Parameters
    ----------
    path : str
        Path to dataset file (.npy format)
    perturbation_moves : int
        Number of perturbation moves per GLS iteration
    iter_limit : int
        Maximum number of GLS iterations
        
    Returns
    -------
    np.ndarray
        Array of tour costs for all instances
    """
    # Load dataset: shape (n_instances, n_cities, 2)
    data = np.load(path)
    n_instances = data.shape[0]
    
    # Generate seeds for reproducibility
    seeds = np.arange(n_instances)
    
    # Process instances with ThreadPoolExecutor for parallel execution
    results = []
    for i in range(n_instances):
        coordinates = data[i]
        result = solve_instance(coordinates, perturbation_moves, iter_limit, int(seeds[i]))
        results.append(result)
    
    return np.array(results)

perturbation_moves_map = {
    50: 30,
    100: 40,
    200: 40,
    500: 50,
}
iter_limit_map = {
    50: 1000,
    100: 1000,
    200: 1000,
    500: 1000,
}

def run(size):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    path = os.path.join(current_dir, 'dataset', f'test_TSP{size}.npy')
    
    # Process dataset
    results = process_file(path, perturbation_moves_map[size], iter_limit_map[size])

    # Add to total cost
    print(results.mean())
    # Calculate optimality gap
    if str(size) in OPTIMAL:
        opt_gap = (results.mean() - OPTIMAL[str(size)]) / OPTIMAL[str(size)] * 100
        print(f"Optimality gap: {opt_gap:.6f}%")

if __name__ == "__main__":
    # Get mode from command line argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run(size)