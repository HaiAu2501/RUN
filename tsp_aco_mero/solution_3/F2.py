import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
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
        # Hyperparameter setup
        alpha = 1.0
        beta = 2.0
        decay_factor = 0.95  # Dynamic decay factor for pheromone levels

        # Apply dynamic decay to pheromone levels
        pheromone = pheromone * decay_factor

        # Sum of attractiveness values
        probabilities = (pheromone ** alpha) * (heuristic ** beta)

        # Handle numerical stability by normalizing the probabilities
        probabilities_sum = probabilities.sum(dim=1, keepdim=True)
        probabilities = probabilities / probabilities_sum

        # Hyperparameter tracking
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities
