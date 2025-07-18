import numpy as np
from F1_final_best import initialize
from F2_final_best import compute_probabilities
from F3_final_best import update_pheromone

def gen_sol(prize, weight, heuristic, pheromone, n_ants, iteration, n_iterations):
    """
    Generate solutions for all ants.
    
    Parameters
    ----------
    prize : np.ndarray, shape (n+1,)
        Prize values (including dummy node).
    weight : np.ndarray, shape (n+1, m)
        Weight matrix (including dummy node).
    heuristic : np.ndarray, shape (n+1,)
        Heuristic values (including dummy node).
    pheromone : np.ndarray, shape (n+1,)
        Pheromone levels.
    n_ants : int
        Number of ants.
    iteration : int
        Current iteration number.
    n_iterations : int
        Total number of iterations.
        
    Returns
    -------
    tuple
        Solutions and their objective values.
    """
    n = prize.shape[0] - 1  # Number of real items (exclude dummy)
    m = weight.shape[1]  # Number of constraints
    
    # Get probabilities using F2
    base_probabilities = compute_probabilities(pheromone, heuristic, iteration, n_iterations)
    
    # Initialize
    solutions = []
    knapsack = np.zeros((n_ants, m))
    mask = np.ones((n_ants, n+1))
    dummy_mask = np.ones((n_ants, n+1))
    dummy_mask[:, -1] = 0  # Dummy node initially unavailable
    
    # Initialize solutions array
    for ant in range(n_ants):
        solutions.append([])
    
    # Update initial feasibility
    mask, knapsack = update_knapsack(mask, knapsack, weight, None, n_ants)
    dummy_mask = update_dummy_state(mask, dummy_mask, n)
    
    # Construction loop
    done = check_done(mask, n)
    while not done:
        # Select next items using base probabilities
        items = pick_item(base_probabilities, mask, dummy_mask, n_ants)
        
        # Add to solutions
        for ant in range(n_ants):
            solutions[ant].append(items[ant])
        
        # Update constraints
        mask, knapsack = update_knapsack(mask, knapsack, weight, items, n_ants)
        dummy_mask = update_dummy_state(mask, dummy_mask, n)
        
        # Check termination
        done = check_done(mask, n)
    
    # Calculate objective values
    objs = np.zeros(n_ants)
    for ant in range(n_ants):
        # For each item in this ant's solution
        for item in solutions[ant]:
            if item < n:  # Skip dummy node
                objs[ant] += prize[item]
    
    return np.array(solutions), objs

def pick_item(base_probabilities, mask, dummy_mask, n_ants):
    """Select next items for all ants based on base probabilities and masks."""
    items = np.zeros(n_ants, dtype=int)
    
    for ant in range(n_ants):
        # Apply masks to base probabilities
        ant_probs = base_probabilities * mask[ant] * dummy_mask[ant]
        
        # Normalize
        sum_probs = ant_probs.sum()
        if sum_probs > 0:
            ant_probs = ant_probs / sum_probs
        else:
            # If no feasible items, select dummy node
            ant_probs = np.zeros_like(ant_probs)
            ant_probs[-1] = 1.0
        
        # Select item
        items[ant] = np.random.choice(len(ant_probs), p=ant_probs)
    
    return items

def check_done(mask, n):
    """Check if all ants are done."""
    return (mask[:, :n] == 0).all()

def update_dummy_state(mask, dummy_mask, n):
    """Enable dummy node when all real items are infeasible."""
    finished = (mask[:, :n] == 0).all(axis=1)
    dummy_mask[finished, -1] = 1
    return dummy_mask

def update_knapsack(mask, knapsack, weight, new_item, n_ants):
    """Update knapsack constraints after item selection."""
    # If items were selected, update mask and capacity
    if new_item is not None:
        for ant in range(n_ants):
            mask[ant, new_item[ant]] = 0
            knapsack[ant] += weight[new_item[ant]]
    
    # Check feasibility of remaining items
    for ant in range(n_ants):
        # Find available items
        candidates = np.where(mask[ant, :-1] > 0)[0]
        
        if len(candidates) > 0:
            # Check if adding each candidate would exceed constraints
            for item in candidates:
                new_capacity = knapsack[ant] + weight[item]
                if (new_capacity > 1.0).any():
                    mask[ant, item] = 0
    
    # Ensure dummy node is always available
    mask[:, -1] = 1
    
    return mask, knapsack

def run_mkp_aco(prize, weight, n_ants=30, n_iterations=100, seed=0):
    """
    Run Ant Colony Optimization for Multiple Knapsack Problem.
    
    Parameters
    ----------
    prize : np.ndarray, shape (n,)
        Prize values for each item.
    weight : np.ndarray, shape (n, m)
        Weight matrix for each item across m constraints.
    n_ants : int
        Number of ants in the colony.
    n_iterations : int
        Number of iterations to run.
        
    Returns
    -------
    float
        Best objective value found.
    """
    # Initialize heuristic and pheromone using F1
    heuristic, pheromone = initialize(prize.copy(), weight.copy())
    
    n, m = weight.shape  # n items, m constraints
    
    # Add dummy node
    prize = np.append(prize, [0.])
    weight = np.append(weight, np.zeros((1, m)), axis=0)
    heuristic = np.append(heuristic, [1e-8])
    pheromone = np.append(pheromone, [1.0])
    
    # Track best solution
    alltime_best_obj = 0
    alltime_best_sol = None
    
    np.random.seed(seed)

    list_objs = []

    # Main optimization loop
    for it in range(n_iterations + 1):
        sols, objs = gen_sol(prize, weight, heuristic, pheromone, n_ants, it, n_iterations)
        
        best_obj, best_idx = objs.max(), objs.argmax()
        
        # Update all-time best
        if best_obj > alltime_best_obj:
            alltime_best_obj = best_obj
            alltime_best_sol = sols[best_idx]
        
        # Update pheromone using F3
        pheromone = update_pheromone(pheromone, sols, objs, it, n_iterations)

        if it % 5 == 0:
            list_objs.append(alltime_best_obj)

    return list_objs

import os
import sys
import numpy as np

def eval_instance(prize, weight, n_ants, n_iter, seed=0):
    if seed is not None:
        np.random.seed(seed)
    
    # Run ACO and return best objective value
    obj = run_mkp_aco(prize, weight, n_ants, n_iter)
    return obj

def process_file(path, n_ants, n_iter):
    # Load the dataset
    data = np.load(path)
    prizes, weights = data['prizes'], data['weights']
    n_instances = prizes.shape[0]
    
    # Generate seeds for reproducibility
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        result = eval_instance(prizes[i], weights[i], n_ants, n_iter, int(seeds[i]))
        results.append(result)
    
    return np.array(results)

def run(size):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    N_ANTS = 10
    N_ITER = 50
    
    path = os.path.join(current_dir, 'datasets', f'test_MKP{size}.npz')
    
    objs = process_file(path, n_ants=N_ANTS, n_iter=N_ITER)
    
    # Add to total
    print(objs.mean(axis=0).tolist())

if __name__ == "__main__":
    # Get mode from command line argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(size)