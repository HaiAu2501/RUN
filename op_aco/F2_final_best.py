# Final round optimized implementation for update_pheromone
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def update_pheromone(pheromone: np.ndarray, sols: list, objs: np.ndarray, it: int, n_iterations: int) -> np.ndarray:
    """
    Innovative pheromone updating mechanism that emphasizes adaptive learning,
    variability in reinforcement based on solution diversity, and a novel exploration
    strategy that opportunistically focuses on high-reward trails.
    """
    # Hyperparameters
    decay = 0.85                # Evaporation factor for pheromone
    max_pheromone = 20.0        # Maximum pheromone value
    min_pheromone = 0.5         # Minimum pheromone value to ensure exploration
    exploration_bias = 5.0       # Emphasizes exploration more aggressively
    diversification_factor = 0.75 # Factor to encourage diverse paths

    # Evaporate existing pheromones
    pheromone *= decay

    total_prizes = np.sum(objs) + 1e-7  # Add small epsilon to avoid division by zero

    # Calculate average performance to determine variance
    avg_performance = np.mean(objs)
    std_performance = np.std(objs)

    # Increase contribution based on performance diversity
    contributions = (objs - (avg_performance - std_performance)) / total_prizes
    contributions = np.clip(contributions, 0, None)  # Ensure non-negative contributions

    # Normalize contributions to scale appropriately
    contributions /= np.sum(contributions) + 1e-7

    # Update pheromone for each ant solution dynamically
    for i, (sol, contrib) in enumerate(zip(sols, contributions)):
        if objs[i] > 0:
            # Dynamic reinforcement based on solution's uniqueness
            reinforcement = contrib * (1 + np.random.rand() * exploration_bias)  # Introduce randomness
            for j in range(len(sol) - 1):
                from_node = sol[j]
                to_node = sol[j + 1]
                pheromone[from_node, to_node] += reinforcement

                # Capping to handle extremes
                pheromone[from_node, to_node] = max(min(pheromone[from_node, to_node], max_pheromone), min_pheromone)

    return pheromone
