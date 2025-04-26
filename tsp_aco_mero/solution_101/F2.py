import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of a probability calculation strategy for ACO TSP with dynamic and adaptive mechanisms.
    """

    def compute(self, pheromone: np.ndarray, heuristic: np.ndarray) -> np.ndarray:
        """
        Compute probabilities for next city selection, incorporating non-linear transformations and adaptive learning.
        
        Args:
            pheromone: Array of shape (n, n) with pheromone levels
            heuristic: Array of shape (n, n) with heuristic values
            
        Returns:
            Array with probability values
        """

        # Hyperparameters for adaptive feedback mechanisms
        alpha = 1.0 + 0.1 * np.log(1 + History.iteration)
        beta = 1.0 + 0.1 * np.log(1 + History.iteration)

        # Non-linear transformations to encourage exploration and diversity
        heuristic_transformed = np.log1p(heuristic)  # Logarithmic transformation
        pheromone_transformed = pheromone ** (alpha)  # Non-linear weighted approach

        # Introduce randomness via a stochastic factor
        stochastic_factor = np.random.rand(*pheromone.shape)  # Random matrix for exploration
        pheromone_transformed *= stochastic_factor  # Modulate pheromone by randomness

        # Combining transformed values with competition effect
        combined_values = pheromone_transformed * heuristic_transformed ** (beta)

        # Normalize the combined values to get valid probabilities
        probabilities = combined_values / np.sum(combined_values)  # Normalize

        # Track hyperparameters for historical analysis
        History.alpha.append(alpha)
        History.beta.append(beta)
        History.pheromone.append(pheromone)
        History.heuristic.append(heuristic)

        return probabilities

