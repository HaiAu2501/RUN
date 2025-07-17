# Final round optimized implementation for update_pheromone
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def update_pheromone(pheromone: np.ndarray, sols: list, objs: np.ndarray, it: int, n_iterations: int) -> np.ndarray:
    """
    Update guidance system based on prize collection performance and exploration balance.
    
    Parameters
    ----------
    pheromone : np.ndarray, shape (n, n)
        Current guidance distribution matrix.
    sols : list
        Ant solution sequences containing node indices.
    objs : np.ndarray, shape (n_ants,)
        Total prize values collected by each ant.
    it : int
        Current optimization iteration index.
    n_iterations : int
        Total planned optimization iterations for adaptive tuning.

    Returns
    -------
    np.ndarray, shape (n, n)
        Updated guidance levels after learning from prize collection quality.
    """
    # Hyperparameters
    decay = 0.9  # Evaporation factor for better retention
    min_pheromone = 1e-5  # Minimum pheromone value
    max_pheromone = 1e3  # Increased maximum to allow stronger paths
    scaling_factor = 4.0  # Moderated scaling for balancing performance reinforcement
    exploration_multiplier = 2.0  # Higher weight on exploration impact per iteration
    adaptive_rate = (n_iterations - it + 1) / n_iterations  # Focus reinforcements on early phases    
    
    # Apply evaporation to pheromone levels
    pheromone *= decay
    pheromone = np.clip(pheromone, min_pheromone, max_pheromone)  # Clamp values
    
    # Calculate performance-based reinforcement scaled by an exploration bonus
    exp_objs = np.exp(objs - np.max(objs))  # Stability adjustment for softmax
    total_exp = np.sum(exp_objs)
    Q = exp_objs / total_exp
    
    # Difference in collected prizes from best to drive exploration
    best_obj = np.max(objs)
    exploration_bonus = np.power(best_obj / (objs + 1e-10), exploration_multiplier)  # Encourage exploring under-performing paths
    
    # Contribution calculation combining performance and exploration reward
    contribution = scaling_factor * Q * exploration_bonus * adaptive_rate
    
    # Update pheromone levels based on contributions from each ant's solution
    for i, sol in enumerate(sols):
        for j in range(len(sol) - 1):
            from_node = sol[j]
            to_node = sol[j + 1]
            pheromone[from_node, to_node] += contribution[i]
            pheromone[to_node, from_node] += contribution[i]  # Bidirectional update
    
    return pheromone
