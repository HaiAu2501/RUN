import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from ga import run_tsp_ga

OPTIMAL = {
    50: 5.698495085905844,
    100: 7.7856204628637276,
    200: 10.67370363366072
}

def eval_instance(coords, population_size, generations, num_runs=3):
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
    num_runs : int
        Number of independent runs to average.
        
    Returns
    -------
    float
        Average of best tour costs across runs.
    """
    # Create distance matrix
    distances = distance_matrix(coords, coords)
    
    # Run GA multiple times and collect results
    results = []
    for run in range(num_runs):
        best_cost = run_tsp_ga(
            distances=distances,
            population_size=population_size,
            generations=generations
        )
        results.append(best_cost)
    
    # Return average cost
    return np.mean(results)

def process_file(path, population_size, generations, num_runs=30):
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
    
    results = []
    for i in range(n_instances):
        coordinates = data[i]
        avg_cost = eval_instance(coordinates, population_size, generations, num_runs)
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
    # Number of runs per instance
    NUM_RUNS = 1
    POPULATION_SIZE = 50
    GENERATIONS = 1000
        
    # Process dataset with multiple runs
    costs = process_file(path, POPULATION_SIZE, GENERATIONS, NUM_RUNS)
    print(f"Average cost for TSP{size} over {NUM_RUNS} runs: {costs.mean():.6f}")
    opt_gap = (costs.mean() - OPTIMAL[size]) / OPTIMAL[size] * 100
    print(f"Optimality gap for TSP{size}: {opt_gap:.6f}%")

if __name__ == "__main__":
    # Get mode from command line argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run(size)