import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP, focused on adaptive normalization
    and stability in hyperparameters for transition probabilities based on pheromone and heuristic values.
    """

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities for next city selection.

        Args:
            pheromone: Tensor of shape (n, n) with pheromone levels
            heuristic: Tensor of shape (n, n) with heuristic values

        Returns:
            Tensor with probability values
        """

        # Fixed values for alpha and beta for stability
        alpha, beta = 1.0, 2.0  # Choose beta > 1 to emphasize heuristic importance

        # Normalize pheromone and heuristic with adaptive normalization
        pheromone_normalized = self.adaptive_normalization(pheromone)
        heuristic_normalized = self.adaptive_normalization(heuristic)

        # Calculate attractiveness values
        probabilities = (pheromone_normalized ** alpha) * (heuristic_normalized ** beta)

        # Normalize to form a valid probability distribution
        probabilities /= (probabilities.sum() + 1e-10)  # Safe to avoid division by zero

        # Track hyperparameters in History for analysis
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities

    def adaptive_normalization(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Adaptive normalization method to ensure robustness against outliers.

        Args:
            tensor: Input tensor to normalize

        Returns:
            Normalized tensor
        """
        mean = torch.mean(tensor) + 1e-5  # Avoid division by zero
        return tensor / mean  # Balance normalization technique: conserve relative values
