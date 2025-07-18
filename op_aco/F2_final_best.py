# Final round optimized implementation for update_pheromone
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def update_pheromone(pheromone: np.ndarray, sols: list, objs: np.ndarray, it: int, n_iterations: int) -> np.ndarray:
    """
    Update guidance system based on prize collection performance with improved dynamic exploration.
    
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
    # Define hyperparameters
    decay = 0.95  # Maintain higher pheromone levels for exploration
    alpha = 3.5   # Slightly reduced reinforcement value for effective exploration
    beta = 0.8    # Maintain high adaptability through enhanced exploration scaling

    # Apply evaporation and clip to avoid low pheromone levels
    pheromone *= decay
    pheromone = np.clip(pheromone, 0.01, None)  # Ensure pheromone never goes below 0.01

    total_obj = np.sum(objs)
    if total_obj <= 0:
        return pheromone
    Q = 1.0 / total_obj  # reinforce based on total performance

    # Identify top performers dynamically, focusing on the top 15% of solutions
    top_count = max(1, int(len(sols) * 0.15))
    top_performers = np.argsort(objs)[-top_count:]

    # Adaptive pheromone contribution calculation
    for idx in top_performers:
        sol = sols[idx]
        obj = objs[idx]

        # Adjust contribution based on relative performance
        contribution = (alpha * obj) * (1 + (it / n_iterations) ** beta)

        # Update pheromone over each edge in the solution path
        for j in range(len(sol) - 1):
            from_node = sol[j]
            to_node = sol[j + 1]
            pheromone[from_node, to_node] += Q * contribution

    # Capping pheromone levels based on maximum observed prize to control over-reinforcement
    max_prize = np.max(objs) if total_obj > 0 else 1.0
    pheromone = np.clip(pheromone, 0, 0.25 * max_prize)  # Higher cap to encourage exploration

    return pheromone