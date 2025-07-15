import numpy as np
from F1 import initialize
from F2 import compute_probabilities
from F3 import update_pheromone

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
    list_costs = []

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

        # Track costs every 10 iterations
        if it % 10 == 0:
            list_costs.append(best_cost)

    assert best_path is not None, "No valid path found"
    assert best_path.shape[0] == n_cities, f"Path length {best_path.shape[0]} != {n_cities}"
    assert len(np.unique(best_path)) == n_cities, "Path contains duplicates"

    return list_costs