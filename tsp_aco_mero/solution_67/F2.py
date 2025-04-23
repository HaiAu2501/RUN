import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities for next city selection.
        
        Args:
            pheromone: Tensor of shape (n, n) with pheromone levels
            heuristic: Tensor of shape (n, n) with heuristic values
        
        Returns:
            Tensor with probability values
        """
        
        # Normalize pheromone and heuristic values to ensure numerical stability
        pheromone_normalized = self.normalize(pheromone)
        heuristic_normalized = self.normalize(heuristic)

        # Compute attractiveness scores
        attractiveness = (pheromone_normalized ** self.alpha) * (heuristic_normalized ** self.beta)

        # Normalize the attractiveness to get valid probability distributions
        probabilities = self.normalize(attractiveness)

        # Record hyperparameters for adaptability tracking
        History.alpha.append(self.alpha)
        History.beta.append(self.beta)

        return probabilities

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor along specified dimensions to achieve numeric stability.
        
        Args:
            tensor: Tensor to be normalized
        
        Returns:
            Normalized tensor
        """
        return tensor / (tensor.sum(dim=1, keepdim=True) + 1e-10)  # Prevent division by zero

    def adapt_parameters(self, performance_metric: float, threshold: float = 0.5):
        """
        Logic for dynamically adapting parameters like alpha and beta.
        Adapt based on a performance metric threshold.
        """
        if performance_metric >= threshold:
            self.alpha *= 1.1  # Increase alpha for greater exploration
        else:
            self.alpha *= 0.95  # Slightly decay alpha for stability
        
        # Record updated alpha for tracking
        History.alpha.append(self.alpha)