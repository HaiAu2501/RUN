import numpy as np

def initialize(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize heuristic and pheromone matrices.

    Parameters
    ----------
    distances : np.ndarray, shape (n_cities, n_cities)
        Matrix of pairwise distances between cities; entries must be positive.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        heuristic : np.ndarray, shape (n_cities, n_cities)
            A matrix representing the desirability of traveling between cities based on local information.
        pheromone : np.ndarray, shape (n_cities, n_cities)
            A matrix representing the initial intensity of pheromone trails, which guide exploration
            based on accumulated experience.
    """
    heuristic = 1.0 / distances 
    pheromone = np.ones_like(distances)  # Initialize pheromone levels uniformly
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
    # Example static parameters; consider tuning externally
    alpha = 1.0  # pheromone influence
    beta = 2.0   # heuristic influence
    return np.power(pheromone, alpha) * np.power(heuristic, beta)

def update_pheromone(
    pheromone: np.ndarray,
    paths: np.ndarray,
    costs: np.ndarray,
    iteration: int,
    n_iterations: int,
) -> np.ndarray:
    """
    Simulate pheromone dynamics: apply evaporation and reinforce edges used by ants.
    Shorter tours contribute more pheromone, guiding future searches.

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
        Updated pheromone levels after evaporation and deposition.
    """
    # Decay factor
    decay = 0.9

    # Evaporation
    pheromone *= decay

    n_cities, n_ants = paths.shape
    # Deposit pheromone: shorter tours deposit more
    for ant in range(n_ants):
        tour = paths[:, ant]
        deposit = 1.0 / costs[ant]
        for i in range(n_cities):
            c = tour[i]
            n = tour[(i + 1) % n_cities]
            pheromone[c, n] += deposit
            pheromone[n, c] += deposit

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
	path = f"datasets/test_TSP{size}.npy"
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