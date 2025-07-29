import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from ga import run_tsp_dr

OPTIMAL = {
    50: 5.698495085905844,
    100: 7.7856204628637276,
    200: 10.67370363366072
}

def eval_instance(coords, seed=None):
    """
    Evaluate a single TSP instance using GA with multiple runs.
    
    Parameters
    ----------
    coords : np.ndarray, shape (n_cities, 2)
        City coordinates.
    population_size : int
        GA population size.
    generations : int
        Number of GA generations.
    seed : int, optional
        Random seed for reproducibility.
    num_runs : int
        Number of independent runs to average.
        
    Returns
    -------
    float
        Average of best tour costs across runs.
    """
    # Create distance matrix
    distances = distance_matrix(coords, coords)

    best_cost = run_tsp_dr(
        distances=distances,
        max_workers=100,
        seed=seed,
    )
    
    # Return average cost
    return best_cost

def process_file(path):
    """
    Process a dataset file and solve all instances with multiple runs.
    
    Parameters
    ----------
    path : str
        Path to dataset file (.npy format).
    population_size : int
        GA population size.
    generations : int
        Number of GA generations.
    num_runs : int
        Number of independent runs per instance.
        
    Returns
    -------
    np.ndarray
        Array of average tour costs for all instances.
    """
    # Load dataset: shape (n_instances, n_cities, 2)
    data = np.load(path)
    n_instances = data.shape[0]
    
    # Generate seeds for reproducibility
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        coordinates = data[i]
        # Use instance index as base seed for reproducibility
        avg_cost = eval_instance(coordinates, seed=int(seeds[i]))
        results.append(avg_cost)
    
    return np.array(results)

def run(size):
    """
    Main evaluation function.
    
    Parameters
    ----------
    mode : str
        Evaluation mode: 'train', 'val', or 'test'.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'datasets', f'test_TSP{size}.npy')
        
    # Process dataset with multiple runs
    costs = process_file(path)
    print(f"Average cost for TSP{size}: {costs.mean():.6f}")
    opt_gap = (costs.mean() - OPTIMAL[size]) / OPTIMAL[size] * 100
    print(f"Optimality gap for TSP{size}: {opt_gap:.6f}%")

if __name__ == "__main__":
    # Get mode from command line argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run(size)