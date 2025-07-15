import numpy as np

def update_pheromone(
    pheromone: np.ndarray,
    paths: np.ndarray,
    costs: np.ndarray,
    iteration: int,
    n_iterations: int,
) -> np.ndarray:
    """
    Update pheromone levels in a way that promotes exploration of less visited routes and refines knowledge of the best routes by using a more dynamic and context-aware approach.
    """
    # Hyperparameters
    decay = 0.95  # Aggressive evaporation to promote exploration
    min_pheromone = 1e-6  # Minimum pheromone level to prevent total evaporation
    max_pheromone = 2.0  # Maximum pheromone level to prevent overload
    quality_scale = 2.5  # Scale factor for determining contribution from paths

    # Evaporation of pheromone levels
    pheromone *= decay
    pheromone[pheromone < min_pheromone] = min_pheromone  # Enforce the minimum limit

    n_cities, n_ants = paths.shape
    total_cost = np.sum(costs)
    average_cost = total_cost / n_ants  # Calculate average cost for normalization
    global_best_cost = np.min(costs)  # Best cost from current iteration

    # Learning from routes using a dynamic scaling mechanism
    # Each ant's tour contributes pheromones in a calibrated manner
    for ant in range(n_ants):
        tour = paths[:, ant]
        cost_ratio = costs[ant] / average_cost  # Ratio used to scale contribution
        contribution = (global_best_cost / costs[ant])**quality_scale if costs[ant] <= global_best_cost else 0.01
        contribution *= (1.0 / (1.0 + cost_ratio))  # Penalize contribution for higher costs

        # Cap the contribution to the maximum limits
        contribution = min(contribution, max_pheromone)

        for i in range(n_cities):
            c = tour[i]
            n = tour[(i + 1) % n_cities]
            pheromone[c, n] += contribution
            pheromone[n, c] += contribution  # Reinforce bi-directional paths

    return pheromone
