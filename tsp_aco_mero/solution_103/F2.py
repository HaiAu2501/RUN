import numpy as np
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values,
    integrating socio-environmental factors and dynamic parameters for enhanced adaptability.
    """

    def compute(self, pheromone: np.ndarray, heuristic: np.ndarray) -> np.ndarray:
        """
        Compute probabilities for next city selection using an adaptive approach.
        Args:
            pheromone: Array of shape (n, n) with pheromone levels
            heuristic: Array of shape (n, n) with heuristic values
        Returns:
            Array with probability values
        """  
        # Dynamic hyperparameters derived from history with increased fluctuation
        alpha = 1.0
        beta = 1.0
        if History.iteration > 0:
            alpha += np.std(History.alpha) * np.random.randn()  # Dynamic adjustment based on std deviation
            beta += np.std(History.beta) * np.random.randn()    # Dynamic adjustment based on std deviation

        # Safe normalization of pheromone and heuristic values to avoid numerical instability
        pheromone_normalized = np.where(pheromone.sum(axis=1, keepdims=True) > 0,
                                         pheromone / pheromone.sum(axis=1, keepdims=True), 0)
        heuristic_normalized = np.where(heuristic.sum(axis=1, keepdims=True) > 0,
                                         heuristic / heuristic.sum(axis=1, keepdims=True), 0)

        # Calculate raw probabilities with adaptive factors and dynamic feedback
        probabilities = (pheromone_normalized ** alpha) * (heuristic_normalized ** beta)

        # Normalize the final probabilities to sum to 1, avoiding division by zero
        probabilities_sum = probabilities.sum(axis=1, keepdims=True)
        probabilities /= np.where(probabilities_sum > 0, probabilities_sum, 1)

        # Hyperparameter tracking with reinforcement of historical context
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities
