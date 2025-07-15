import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from concurrent.futures import ThreadPoolExecutor
from aco import run_tsp_aco

def eval_instance(coords, n_ants, n_iter, seed):
    D = distance_matrix(coords, coords)
    return run_tsp_aco(D, n_ants, n_iter, seed)

def process_file(path, n_ants, n_iter):
    data = np.load(path)  # shape (n_instances, size, 2)
    n_instances = data.shape[0]
    seeds = np.arange(n_instances)
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(eval_instance, data[i], n_ants, n_iter, int(seeds[i]))
            for i in range(n_instances)
        ]
        for f in futures:
            results.append(f.result())
    return np.array(results)

def main(size):
    print(f"Processing TSP instances of size {size}...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'datasets', f'test_TSP{size}.npy')
    list_costs = process_file(path, n_ants=50, n_iter=100)
    mean_cost = np.mean(list_costs, axis=0) # Average costs over all instances
    print(mean_cost.tolist())

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(size)
