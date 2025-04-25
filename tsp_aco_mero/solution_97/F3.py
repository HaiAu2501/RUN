import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of adaptive pheromone update strategy for ACO TSP using dynamic exploration with temperature regulation.
    """
    def update(self, pheromone: np.ndarray, paths: np.ndarray, costs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Update pheromone levels based on ant paths and solution costs with a temperature-modulated exploration.

        Args:
            pheromone: Array of shape (n, n) with current pheromone levels
            paths: Array of shape (n_cities, n_ants) with paths taken by ants
            costs: Array of shape (n_ants,) with path costs
            temperature: Current exploration temperature

        Returns:
            Updated pheromone array
        """
        # Parameters
        decay = 0.9
        alpha = 2.0

        # Decay existing pheromones
        pheromone *= decay

        n_ants = paths.shape[1]

        # Non-linear transformation with temperature-modulated exploration
        costs_normalized = costs - np.min(costs) + 1e-6  # Avoid division by zero
        rewards = np.exp(-costs_normalized / temperature)  # Reward inversely proportional to normalized costs, adjusted by temperature
        rewards /= np.sum(rewards)  # Normalize rewards

        for i in range(n_ants):
            path = paths[:, i]
            success_score = rewards[i]  # Reinforcement from successful paths
            next_cities = np.roll(path, -1)  # Get edges for pheromone update

            # Update pheromone for the explored path
            for j in range(len(path)):
                current_city = path[j]
                next_city = next_cities[j]

                # Apply temperature-based factor to reinforce exploration
                pheromone[current_city, next_city] += (alpha * success_score * (1 + np.random.rand() / temperature))
                pheromone[next_city, current_city] += (alpha * success_score * (1 + np.random.rand() / temperature))  # Symmetric updates

        # Track history state for potential performance metrics
        History.decay.append(decay)
        History.alpha.append(alpha)
        History.pheromone.append(pheromone)

        return pheromone
