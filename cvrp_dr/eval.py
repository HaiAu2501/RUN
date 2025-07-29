import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from dr import run_cvrp_dr

# Problem constants matching CVRP_ACO
CAPACITY = 50

def eval_instance(coords, demands, capacity, seed=None):
    """
    Evaluate a single CVRP instance using Deconstruction-Repair.
    
    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Node coordinates (depot at index 0).
    demands : np.ndarray, shape (n,)
        Node demands (depot has demand 0).
    capacity : int
        Vehicle capacity constraint.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    float
        Total routing cost found.
    """
    # Create distance matrix
    distances = distance_matrix(coords, coords)
    
    # Run Deconstruction-Repair algorithm
    best_cost = run_cvrp_dr(
        distances=distances,
        demands=demands,
        capacity=capacity,
        destruction_rate=0.3,
        use_2opt=False,
        max_workers=20,
        seed=seed
    )
    
    return best_cost

def process_file(path):
    """
    Process a dataset file and solve all instances.
    
    Parameters
    ----------
    path : str
        Path to dataset file (.npy format).
        
    Returns
    -------
    np.ndarray
        Array of routing costs for all instances.
    """
    # Load dataset: shape (n_instances, n_nodes, 3) where last dim is [demand, x, y]
    dataset = np.load(path)
    demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
    
    n_instances = node_positions.shape[0]
    
    # Generate seeds for reproducibility
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        coordinates = node_positions[i]
        instance_demands = demands[i]
        
        # Use instance index as seed for reproducibility
        cost = eval_instance(coordinates, instance_demands, CAPACITY, seed=int(seeds[i]))
        results.append(cost)
    
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
    
    path = os.path.join(current_dir, 'datasets', f'test_CVRP{size}.npy')
    
    # Process dataset
    costs = process_file(path)
    
    # Print total cost
    print(costs.mean())

if __name__ == "__main__":
    # Get mode from command line argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(size)