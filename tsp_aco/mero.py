import numpy as np

def initialize(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_cities = distances.shape[0]

    # Initialize heuristic using nonlinear relationships and dynamic factors
    heuristic = np.zeros_like(distances)
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                # Non-linear attractiveness based on distance
                attractiveness = (1 / (distances[i, j] ** 1.5)) + np.random.rand() * 0.1
                heuristic[i, j] = attractiveness ** 2  # Square it for non-linear scaling

    # Normalize heuristic to ensure comparable scale
    heuristic = heuristic / heuristic.sum(axis=1, keepdims=True)

    # Initialize pheromone levels that dynamically adapt over iterations
    pheromone = np.full((n_cities, n_cities), 0.1)  # Start with a small amount of pheromone
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                # Adjust pheromones based on distance and adaptability factor
                pheromone[i, j] = 1 / distances[i, j] ** 2 + np.random.rand() * 0.05

    # Normalize pheromones to allow for comparative measures
    pheromone = pheromone / pheromone.sum(axis=1, keepdims=True)

    return heuristic, pheromone


def compute_probabilities(
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    iteration: int,
    n_iterations: int
) -> np.ndarray:
    """
    Generate unnormalized transition weights reflecting pheromone intensity and heuristic desirability.
    These weights determine the likelihood of ants moving between cities.

    Parameters
    ----------
    pheromone : np.ndarray, shape (n_cities, n_cities)
        Current pheromone levels on edges.
    heuristic : np.ndarray, shape (n_cities, n_cities)
        Heuristic desirability of edges.
    iteration : int
        Index of the current iteration.
    n_iterations : int
        Total number of iterations for adaptive tuning.

    Returns
    -------
    np.ndarray, shape (n_cities, n_cities)
        Unnormalized transition weights for ant decisions.
    """
    # Dynamic parameters for runtime adaptation
    alpha = 1  # Pheromone influence factor
    beta = 1   # Heuristic influence factor
    decay = 0.9  # Non-linear pheromone decay
    exploration_rate = 1 - (iteration / (n_iterations + 1)) 

    # Weighted pheromone with non-linear decay based on performance
    pheromone_weighted = np.power(pheromone, alpha) * (decay ** (n_iterations - iteration))

    # Heuristic weights emphasizing desirability
    heuristic_weighted = np.power(heuristic, beta)

    # Combine pheromone and heuristic weightings with exploration
    combined_weights = pheromone_weighted * heuristic_weighted * exploration_rate

    # Normalize combined weights to get transition probabilities
    transition_probabilities = combined_weights / np.sum(combined_weights, axis=1, keepdims=True)

    return transition_probabilities


def update_pheromone(
    pheromone: np.ndarray,
    paths: np.ndarray,
    costs: np.ndarray,
    iteration: int,
    n_iterations: int,
) -> np.ndarray:
    """
    Simulate adaptive pheromone dynamics with exploration-enhancing and qualitative reinforcement.

    Parameters
    ----------
    pheromone : np.ndarray, shape (n_cities, n_cities)
        Current pheromone distribution.
    paths : np.ndarray, shape (n_cities, n_ants)
        Ant tour sequences of city indices per column.
    costs : np.ndarray, shape (n_ants,)
        Total tour cost for each ant.
    iteration : int
        Index of the current iteration.
    n_iterations : int
        Total number of iterations for adaptive tuning.

    Returns
    -------
    np.ndarray, shape (n_cities, n_cities)
        Updated pheromone levels after evaporation and dynamic deposition.
    """
    # Dynamic evaporation based on iteration and costs (inverse)
    decay_factor = 0.95 - 0.15 * (np.mean(costs) / (np.max(costs) + 1e-10))
    pheromone *= decay_factor

    # Normalize costs and calculate normalized pheromone deposit based on distances
    best_cost_indices = np.argsort(costs)[:int(len(costs)*0.5)]  # Top 50% of ants' paths
    deposit_strength = np.zeros_like(pheromone)

    for ant in best_cost_indices:
        tour = paths[:, ant]
        norm_cost = 1.0 / (costs[ant] + 1e-10)  # Normalize individual ant's cost
        for i in range(len(tour)):
            c = tour[i]
            n = tour[(i + 1) % len(tour)]
            pheromone[c, n] += norm_cost  # Deposition based on cost
            pheromone[n, c] += norm_cost  # Ensure symmetry

    # Introduce randomness by applying thermal noise-like adjustments
    if iteration % (n_iterations // 5) == 0:
        noise_strength = np.random.uniform(0.1, 0.3, size=pheromone.shape)
        pheromone += noise_strength * (np.random.rand(*pheromone.shape) - 0.5)  # Random perturbations

    # Limit pheromone values to avoid too high accumulation while retaining diversity
    pheromone = np.clip(pheromone, 0, np.max(pheromone) * 1.5)
    return pheromone


def constructive_path(n_cities, n_ants, probabilities, seed=None):
    """
    Construct tours for all ants given transition probabilities.
    Sampling from categorical distributions per ant (vectorized, like torch.Categorical).
    """
    rng = np.random.default_rng(seed)
    start = rng.integers(n_cities, size=n_ants)
    paths = np.empty((n_cities, n_ants), dtype=int)
    paths[0] = start

    mask = np.ones((n_ants, n_cities), dtype=float)
    mask[np.arange(n_ants), start] = 0.0

    current = start
    for step in range(1, n_cities):
        probs = probabilities[current] * mask
        sums = probs.sum(axis=1)
        valid = sums > 0.0
        probs[valid] = probs[valid] / sums[valid][:, None]
        cum = np.cumsum(probs, axis=1)
        r = rng.random(n_ants)
        next_city = np.argmax(cum > r[:, None], axis=1)

        invalid = ~valid
        for ant in np.nonzero(invalid)[0]:
            unv = np.where(mask[ant] > 0.0)[0]
            if unv.size > 0:
                next_city[ant] = rng.choice(unv)
            else:
                next_city[ant] = rng.integers(n_cities)

        paths[step] = next_city
        mask[np.arange(n_ants), next_city] = 0.0
        current = next_city

    return paths

def calculate_path_costs(paths, distances):
    """
    Vectorized path cost calculation.
    """
    cost_steps = distances[paths[:-1], paths[1:]]
    return_back = distances[paths[-1], paths[0]]
    return cost_steps.sum(axis=0) + return_back

def run_tsp_aco(distances, n_ants=50, n_iterations=100, seed=0):
    """
    Execute ACO and return best cost, with assertions on path validity.
    """
    distances = np.asarray(distances, dtype=float)
    distances[distances < 1e-6] = 1e-6

    n_cities = distances.shape[0]
    heuristic, pheromone = initialize(distances.copy())

    best_cost = np.inf
    best_path = None
    rng_seed = seed

    cost_list = []

    for it in range(n_iterations + 1):
        probs = compute_probabilities(pheromone, heuristic, it, n_iterations)
        paths = constructive_path(n_cities, n_ants, probs, seed=rng_seed)
        costs = calculate_path_costs(paths, distances)

        idx = np.argmin(costs)
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best_path = paths[:, idx].copy()

        pheromone = update_pheromone(pheromone, paths, costs, it, n_iterations)
        rng_seed += 1

        if it % 20 == 0:
            cost_list.append(best_cost)

    assert best_path is not None, "No valid path found"
    assert best_path.shape[0] == n_cities, f"Path length {best_path.shape[0]} != {n_cities}"
    assert len(np.unique(best_path)) == n_cities, "Path contains duplicates"

    return best_cost, cost_list

import sys
from scipy.spatial import distance_matrix

size = sys.argv[1]

if size in ["20", "50", "100"]:
	N_ANTS = 50
	N_ITERATIONS = 200
else:
	N_ANTS = 100
	N_ITERATIONS = 200

def run_aco(seed=0):
	costs = []
	path = f"tsp_aco/datasets/test_TSP{size}.npy"
	batch = np.load(path)
	for i, prob in enumerate(batch):
		distances = distance_matrix(prob, prob)
		obj, cost_list = run_tsp_aco(distances=distances, n_ants=N_ANTS, n_iterations=N_ITERATIONS)
		costs.append(cost_list)
	mean_costs = np.mean(np.array(costs), axis=0)
	print(mean_costs.tolist())

if __name__ == "__main__":
    print(f"Running ACO for TSP{size} with n_ants={N_ANTS} and n_iterations={N_ITERATIONS}")
    run_aco(seed=0)