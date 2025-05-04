import numpy as np

def initialize(prize: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize heuristic and pheromone matrices for Multiple Knapsack Problem.
    
    Parameters
    ----------
    prize : np.ndarray, shape (n,)
        Prize values for each item.
    weight : np.ndarray, shape (n, m)
        Weight matrix for each item across m constraints.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        heuristic : np.ndarray, shape (n,)
            Heuristic values for each item based on adaptable metrics.
        pheromone : np.ndarray, shape (n,)
            Adaptive pheromone levels based on initial item desirability.
    """
    n, m = weight.shape
    
    # Calculate the average weight across constraints for each item
    avg_weight = np.mean(weight, axis=1)
    
    # Calculate heuristic values based on a modified prize-to-weight ratio
    heuristic = prize / (avg_weight + 1e-10)  # Prevent division by zero
    
    # Introduce a variance in pheromone initialization based on prize values
    pheromone = np.exp(prize - np.max(prize))  # Normalize by exp for smoother values
    pheromone /= np.sum(pheromone)  # Normalize to sum to 1
    
    return heuristic, pheromone


def update_pheromone(pheromone: np.ndarray, sols: np.ndarray, objs: np.ndarray) -> np.ndarray:
    """
    Update pheromone levels based on solutions for the Multiple Knapsack Problem.
    
    Parameters
    ----------
    pheromone : np.ndarray, shape (n+1,)
        Current pheromone levels (including dummy node).
    sols : np.ndarray
        Solutions constructed by ants, containing indices of selected items.
    objs : np.ndarray, shape (n_ants,)
        Objective values (total prize) of solutions.
        
    Returns
    -------
    np.ndarray, shape (n+1,)
        Updated pheromone levels.
    """  

    # Evaporation with chaotic effects for exploration - inject randomness in the decay
    decay_base = 0.85
    chaos_factor = 0.1 * np.random.rand(len(pheromone))
    decay = decay_base + chaos_factor
    pheromone = pheromone * decay
    pheromone[pheromone < 0] = 0

    # Calculate an adaptive pheromone deposit rate based on rankings of objective values
    sorted_indices = np.argsort(objs)[::-1]
    rank = np.argsort(sorted_indices) + 1  # Rank from 1 to n_ants
    deposit_factor = 1 / (rank + 1e-10)  # Avoid division by zero

    # Multi-dimensional exploration: Combine objective value and diversity
    unique_items_count = np.array([len(np.unique(s)) for s in sols])
    total_quality = objs * unique_items_count
    total_weighted_quality = np.sum(total_quality)

    # Balance pheromone deposition across all solutions based on a chosen metric
    total_inv_quality = np.sum(1 / (total_quality + 1e-10))
    adjust_rate = 1.0 / (total_inv_quality if total_inv_quality > 0 else 1)

    for i in range(len(objs)):
        sol = sols[i]
        # Deposit pheromone influenced by both quality and diversity
        pheromone[sol] += adjust_rate * (total_quality[i] / total_weighted_quality) * deposit_factor[i]

    return pheromone


def gen_sol(prize, weight, heuristic, pheromone, n_ants, alpha, beta):
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
    alpha : float
        Pheromone influence parameter.
    beta : float
        Heuristic influence parameter.
        
    Returns
    -------
    tuple
        Solutions and their objective values.
    """
    n = prize.shape[0] - 1  # Number of real items (exclude dummy)
    m = weight.shape[1]  # Number of constraints
    
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
        # Select next items
        items = pick_item(pheromone, heuristic, mask, dummy_mask, n_ants, alpha, beta, seed=0)
        
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

def pick_item(pheromone, heuristic, mask, dummy_mask, n_ants, alpha, beta, seed=None):
    """Select next items for all ants."""
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate probabilities
    probs = np.zeros_like(mask)
    for ant in range(n_ants):
        # Calculate unnormalized probabilities
        probs[ant] = (pheromone ** alpha) * (heuristic ** beta) * mask[ant] * dummy_mask[ant]
        
        # Normalize
        sum_probs = probs[ant].sum()
        if sum_probs > 0:
            probs[ant] = probs[ant] / sum_probs
        else:
            # If no feasible items, select dummy node
            probs[ant, -1] = 1.0
    
    # Select items
    items = np.zeros(n_ants, dtype=int)
    for ant in range(n_ants):
        items[ant] = np.random.choice(len(pheromone), p=probs[ant])
    
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

def run_mkp_aco(prize, weight, n_ants=30, n_iterations=100):
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
    heuristic, pheromone = initialize(prize, weight)
    
    n, m = weight.shape  # n items, m constraints
    
    # Add dummy node
    prize = np.append(prize, [0.])
    weight = np.append(weight, np.zeros((1, m)), axis=0)
    heuristic = np.append(heuristic, [1e-8])
    pheromone = np.append(pheromone, [1.0])
    
    # Parameters
    alpha = 1  # pheromone influence
    beta = 1   # heuristic influence
    
    # Track best solution
    alltime_best_obj = 0
    alltime_best_sol = None
    
    # Main optimization loop
    for _ in range(n_iterations):
        sols, objs = gen_sol(prize, weight, heuristic, pheromone, n_ants, alpha, beta)
        
        best_obj, best_idx = objs.max(), objs.argmax()
        
        # Update all-time best
        if best_obj > alltime_best_obj:
            alltime_best_obj = best_obj
            alltime_best_sol = sols[best_idx]
        
        # Update pheromone
        pheromone = update_pheromone(pheromone, sols, objs)
    
    # Verify solution validity
    assert alltime_best_sol is not None, "No valid solution found"
    
    # Check if solution is feasible
    if len(alltime_best_sol) > 0:
        # Filter out dummy node
        actual_items = [item for item in alltime_best_sol if item < n]
        
        # Check if solution respects all constraints
        total_weight = np.zeros(m)
        for item in actual_items:
            total_weight += weight[item]
        assert (total_weight <= 1.0).all(), "Solution violates constraints"
        
        # Verify objective value matches solution
        calculated_obj = sum(prize[item] for item in actual_items)
        assert np.isclose(calculated_obj, alltime_best_obj), "Objective value doesn't match solution"
    
    return alltime_best_obj

