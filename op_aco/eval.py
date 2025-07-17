import numpy as np
from F1_final_best import initialize
from F2_final_best import update_pheromone

def add_dummy_node(prizes, distances, heuristic, pheromone):
    """
    Add a dummy node to the problem data.
    
    Parameters
    ----------
    prizes : np.ndarray, shape (n,)
        Prize values for each node.
    distances : np.ndarray, shape (n, n)
        Distance matrix between nodes.
    heuristic : np.ndarray, shape (n, n)
        Heuristic matrix.
    pheromone : np.ndarray, shape (n, n)
        Pheromone matrix.
        
    Returns
    -------
    tuple
        Data with dummy node added.
    """
    n = prizes.shape[0]
    
    # Add dummy node to prizes (with very small prize value)
    prizes_new = np.append(prizes, 1e-10)
    
    # Add dummy node to distances
    # Create a row with large values (cannot go from dummy to any node)
    new_row = np.full((1, n), 1e10)
    distances_new = np.vstack((distances, new_row))
    
    # Create a column with small values (can go to dummy with minimal cost)
    new_col = np.full((n+1, 1), 1e-10)
    distances_new = np.hstack((distances_new, new_col))
    
    # Add dummy node to heuristic
    # Cannot reach other nodes from dummy node
    new_row = np.zeros((1, n))
    heuristic_new = np.vstack((heuristic, new_row))
    
    # All nodes can reach dummy node with high desirability
    new_col = np.ones((n+1, 1))
    heuristic_new = np.hstack((heuristic_new, new_col))
    
    # Create new pheromone matrix with dummy node
    pheromone_new = np.ones_like(distances_new)
    
    # Set dummy node special values
    distances_new[distances_new == 1e-10] = 0
    prizes_new[-1] = 0
    
    return prizes_new, distances_new, heuristic_new, pheromone_new

def pick_node(pheromone, heuristic, mask, dummy_mask, cur_node, n_ants, alpha, beta):
    """
    Select next node for each ant based on pheromone and heuristic values.
    
    Parameters
    ----------
    pheromone : np.ndarray, shape (n+1, n+1)
        Pheromone levels.
    heuristic : np.ndarray, shape (n+1, n+1)
        Heuristic values.
    mask : np.ndarray, shape (n_ants, n+1)
        Feasibility mask.
    dummy_mask : np.ndarray, shape (n_ants, n+1)
        Dummy node mask.
    cur_node : np.ndarray, shape (n_ants,)
        Current node of each ant.
    n_ants : int
        Number of ants.
    alpha : float
        Pheromone influence factor.
    beta : float
        Heuristic influence factor.
        
    Returns
    -------
    np.ndarray
        Next node for each ant.
    """
    next_node = np.zeros(n_ants, dtype=int)
    
    for ant in range(n_ants):
        # Get combined mask (feasible nodes considering dummy node availability)
        combined_mask = mask[ant] * dummy_mask[ant]
        
        if np.sum(combined_mask) == 0:
            # No feasible nodes, pick dummy node if available
            if dummy_mask[ant, -1] == 1:
                next_node[ant] = len(combined_mask) - 1
            else:
                next_node[ant] = 0  # Fallback to depot
            continue
            
        # Get current node
        cur = cur_node[ant]
        
        # Calculate transition probabilities
        ph = pheromone[cur]
        heu = heuristic[cur]
        prob = (ph ** alpha) * (heu ** beta) * combined_mask
        
        # Normalize probabilities
        sum_prob = np.sum(prob)
        if sum_prob > 0:
            prob = prob / sum_prob
        else:
            # No feasible choices
            next_node[ant] = 0
            continue
        
        # Choose next node based on probabilities
        next_node[ant] = np.random.choice(len(ph), p=prob)
    
    return next_node

def update_mask(travel_dis, cur_node, mask, distances, max_len, n_ants):
    """
    Update feasibility mask based on travel distance constraints.
    
    Parameters
    ----------
    travel_dis : np.ndarray, shape (n_ants,)
        Travel distance for each ant.
    cur_node : np.ndarray, shape (n_ants,)
        Current node of each ant.
    mask : np.ndarray, shape (n_ants, n+1)
        Feasibility mask.
    distances : np.ndarray, shape (n+1, n+1)
        Distance matrix.
    max_len : float
        Maximum allowed tour length.
    n_ants : int
        Number of ants.
        
    Returns
    -------
    np.ndarray
        Updated feasibility mask.
    """
    # Remove current node from available choices
    for ant in range(n_ants):
        mask[ant, cur_node[ant]] = 0
    
    # Check feasibility based on distance constraint
    n = mask.shape[1] - 1  # Exclude dummy node
    
    for ant in range(n_ants):
        if cur_node[ant] != n:  # If not at dummy node
            # Find candidate nodes
            candidates = np.where(mask[ant, :n] > 0)[0]
            if len(candidates) > 0:
                # Check if after going to candidate, can return to depot
                for node in candidates:
                    # Calculate total distance: current path + to candidate + back to depot
                    new_dist = travel_dis[ant] + distances[cur_node[ant], node] + distances[node, 0]
                    if new_dist > max_len:
                        mask[ant, node] = 0  # Mark as infeasible if exceeds max length
    
    # Mask dummy node for all ants (updated in update_dummy_state)
    mask[:, -1] = 0
    
    return mask

def update_dummy_state(mask, dummy_mask, n):
    """
    Update dummy node availability.
    
    Parameters
    ----------
    mask : np.ndarray, shape (n_ants, n+1)
        Feasibility mask.
    dummy_mask : np.ndarray, shape (n_ants, n+1)
        Dummy node mask.
    n : int
        Number of real nodes (excluding dummy).
        
    Returns
    -------
    np.ndarray
        Updated dummy node mask.
    """
    # Enable dummy node when all real nodes are infeasible
    all_real_masked = (mask[:, :n] == 0).all(axis=1)
    dummy_mask[all_real_masked, -1] = 1
    
    return dummy_mask

def check_done(mask, dummy_mask, n):
    """
    Check if ants have completed their tours.
    
    Parameters
    ----------
    mask : np.ndarray, shape (n_ants, n+1)
        Feasibility mask.
    dummy_mask : np.ndarray, shape (n_ants, n+1)
        Dummy node mask.
    n : int
        Number of real nodes (excluding dummy).
        
    Returns
    -------
    np.ndarray
        Boolean array indicating which ants have completed their tours.
    """
    # An ant is done when all real nodes are masked and dummy node is available
    real_nodes_masked = (mask[:, :n] == 0).all(axis=1)
    dummy_available = dummy_mask[:, -1] == 1
    return real_nodes_masked & dummy_available

def gen_sol_obj(solutions, prizes):
    """
    Calculate objective values for solutions.
    
    Parameters
    ----------
    solutions : list
        List of solutions.
    prizes : np.ndarray, shape (n+1,)
        Prize values including dummy node.
        
    Returns
    -------
    np.ndarray
        Objective values of solutions.
    """
    n_ants = len(solutions)
    objs = np.zeros(n_ants)
    
    for ant in range(n_ants):
        # Sum prizes for all unique nodes in the solution
        unique_nodes = np.unique(solutions[ant])
        for node in unique_nodes:
            if node < prizes.shape[0] - 1:  # Skip dummy node
                objs[ant] += prizes[node]
    
    return objs

def gen_sol(prizes, distances, max_len, heuristic, pheromone, n_ants, alpha, beta):
    """
    Generate solutions for all ants.
    
    Parameters
    ----------
    prizes : np.ndarray, shape (n+1,)
        Prize values including dummy node.
    distances : np.ndarray, shape (n+1, n+1)
        Distance matrix including dummy node.
    max_len : float
        Maximum allowed tour length.
    heuristic : np.ndarray, shape (n+1, n+1)
        Heuristic values.
    pheromone : np.ndarray, shape (n+1, n+1)
        Pheromone levels.
    n_ants : int
        Number of ants.
    alpha : float
        Pheromone influence factor.
    beta : float
        Heuristic influence factor.
        
    Returns
    -------
    tuple
        Solutions and their objective values.
    """
    n = prizes.shape[0] - 1  # Exclude dummy node
    
    # Initialize solutions
    solutions = []
    for _ in range(n_ants):
        solutions.append([0])  # All ants start at depot (node 0)
    
    # Initialize masks
    mask = np.ones((n_ants, n+1))
    mask[:, 0] = 0  # Cannot return to depot immediately
    
    # Dummy node is not available initially
    dummy_mask = np.ones((n_ants, n+1))
    dummy_mask[:, -1] = 0
    
    # Initialize tracking variables
    all_done = np.zeros(n_ants, dtype=bool)
    travel_dis = np.zeros(n_ants)
    cur_node = np.zeros(n_ants, dtype=int)
    
    # Main construction loop
    while not np.all(all_done):
        # Update masks based on current state
        mask = update_mask(travel_dis, cur_node, mask, distances, max_len, n_ants)
        dummy_mask = update_dummy_state(mask, dummy_mask, n)
        
        # Check if ants are done
        all_done = check_done(mask, dummy_mask, n)
        
        if np.all(all_done):
            break
        
        # Pick next node for each ant
        nxt_node = pick_node(pheromone, heuristic, mask, dummy_mask, cur_node, n_ants, alpha, beta)
        
        # Update solutions and tracking variables
        for ant in range(n_ants):
            if not all_done[ant]:
                solutions[ant].append(nxt_node[ant])
                travel_dis[ant] += distances[cur_node[ant], nxt_node[ant]]
                cur_node[ant] = nxt_node[ant]
    
    # Calculate objective values
    objs = gen_sol_obj(solutions, prizes)
    
    return solutions, objs

def run_op_aco(prizes, distances, max_len, n_ants=20, n_iterations=50, alpha=1, beta=1):
    """
    Run Ant Colony Optimization for Orienteering Problem.
    
    Parameters
    ----------
    prizes : np.ndarray, shape (n,)
        Prize values for each node.
    distances : np.ndarray, shape (n, n)
        Distance matrix between nodes.
    max_len : float
        Maximum allowed tour length.
    n_ants : int, optional
        Number of ants, by default 20
    n_iterations : int, optional
        Number of iterations, by default 50
    alpha : float, optional
        Pheromone influence factor, by default 1
    beta : float, optional
        Heuristic influence factor, by default 1
    decay : float, optional
        Pheromone evaporation rate, by default 0.9
        
    Returns
    -------
    float
        Best objective value found.
    list
        Best solution found.
    """
    # Initialize heuristic and pheromone using F1
    heuristic, pheromone = initialize(prizes, distances.copy(), max_len)
    
    # Add dummy node
    prizes_new, distances_new, heuristic_new, pheromone_new = add_dummy_node(prizes, distances, heuristic, pheromone)
    
    # Track best solution
    best_obj = 0
    best_sol = None

    list_obj = []
    
    # Main optimization loop
    for it in range(n_iterations + 1):
        # Generate solutions
        sols, objs = gen_sol(prizes_new, distances_new, max_len, heuristic_new, pheromone_new, n_ants, alpha, beta)
        
        # Find best solution in this iteration
        if len(objs) > 0:
            iter_best_idx = np.argmax(objs)
            iter_best_obj = objs[iter_best_idx]
            iter_best_sol = sols[iter_best_idx]
            
            # Update all-time best solution
            if iter_best_obj > best_obj:
                best_obj = iter_best_obj
                best_sol = iter_best_sol
        
        # Update pheromone levels using F2
        pheromone_new = update_pheromone(pheromone_new, sols, objs, it, n_iterations)

        if it % 10 == 0:
            list_obj.append(best_obj)

    return list_obj

############################ EVAL SCRIPT ############################

import os, sys

def gen_prizes(coordinates):
    """
    Generate prize values based on distance from depot.
    
    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 2)
        Node coordinates.
        
    Returns
    -------
    np.ndarray
        Prize values.
    """
    depot_coor = coordinates[0]
    distances = np.sqrt(np.sum((coordinates - depot_coor)**2, axis=1))
    prizes = 1 + np.floor(99 * distances / distances.max())
    prizes = prizes / prizes.max()
    return prizes

def gen_distance_matrix(coordinates):
    """
    Generate distance matrix from coordinates.
    
    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 2)
        Node coordinates.
        
    Returns
    -------
    np.ndarray
        Distance matrix.
    """
    n_nodes = len(coordinates)
    distances = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                distances[i, j] = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
    
    # Set diagonal to high value to prevent self-loops
    np.fill_diagonal(distances, 1e9)
    
    return distances

def get_max_len(n):
    """
    Get maximum tour length based on problem size.
    
    Parameters
    ----------
    n : int
        Problem size.
        
    Returns
    -------
    float
        Maximum tour length.
    """
    threshold_list = [50, 100, 200, 300]
    maxlen = [3.0, 4.0, 5.0, 6.0]
    for threshold, result in zip(threshold_list, maxlen):
        if n <= threshold:
            return result
    return 7.0

def create_instance(coordinates, idx=0):
    """
    Create an OP instance from coordinates.
    
    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 2)
        Node coordinates.
    idx : int
        Instance index.
        
    Returns
    -------
    tuple
        (prizes, distances, maxlen)
    """
    n = coordinates.shape[0]
    
    # Create distance matrix
    distances = gen_distance_matrix(coordinates)
    
    # Generate prizes
    prizes = gen_prizes(coordinates)
    
    # Get maximum tour length
    maxlen = get_max_len(n)
    
    return prizes, distances, maxlen

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
        Result objectives.
    """
    # Load the dataset
    data = np.load(path)
    coordinates = data['coordinates']
    n_instances = coordinates.shape[0]
    
    # Generate seeds for reproducibility
    seeds = np.arange(n_instances)
    
    # Process all instances in parallel
    results = []
    for i in range(n_instances):
        prizes, distances, maxlen = create_instance(coordinates[i])
        results.append(eval_instance(prizes, distances, maxlen, n_ants, n_iter, int(seeds[i])))
    
    return np.array(results)

def eval_instance(prizes, distances, maxlen, n_ants, n_iter, seed=0):
    """
    Evaluate a single instance.
    
    Parameters
    ----------
    prizes : np.ndarray
        Prize values.
    distances : np.ndarray
        Distance matrix.
    maxlen : float
        Maximum tour length.
    n_ants : int
        Number of ants.
    n_iter : int
        Number of iterations.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    float
        Best objective value.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Run ACO and return best objective value
    obj: list = run_op_aco(prizes, distances, maxlen, n_ants, n_iter)
    return obj

def run(size):
    print(f"Running OP-ACO-MOTIF for OP{size}...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'datasets', f'test_OP{size}.npz')
    
    N_ANTS = 20
    N_ITER = 100
    
    res = process_file(path, n_ants=N_ANTS, n_iter=N_ITER)
    print(res.mean(axis=0).tolist())

if __name__ == "__main__":
    # Get mode from command line argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(size)