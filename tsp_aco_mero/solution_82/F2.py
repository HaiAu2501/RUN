import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of the probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha  # Weight for pheromone
        self.beta = beta    # Weight for heuristic

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities for next city selection.
        
        Args:
            pheromone: Tensor of shape (n, n) with pheromone levels
            heuristic: Tensor of shape (n, n) with heuristic values
            
        Returns:
            Tensor with probability values
        """
        # Calculate attractiveness values by combining pheromone and heuristic information
        attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)

        # Improved normalization for stability
        total_attractiveness = attractiveness.sum(dim=1, keepdim=True) + 1e-10
        probabilities = attractiveness / total_attractiveness

        # Track hyperparameters history
        self.track_history()

        return probabilities
    
    def update_parameters(self, alpha: float, beta: float):
        """
        Update alpha and beta values for dynamic adaptation.
        """
        alpha_update = self.smooth_update(self.alpha, alpha)
        beta_update = self.smooth_update(self.beta, beta)

        # Update parameters only if their values are valid
        if self.is_valid(alpha_update) and self.is_valid(beta_update):
            self.alpha = alpha_update
            self.beta = beta_update

    def smooth_update(self, current: float, new: float) -> float:
        """
        Smoothly update the current value with a new one.
        """
        return 0.9 * current + 0.1 * new

    def is_valid(self, value: float) -> bool:
        """
        Check if the parameter value is valid.
        """
        return value > 0

    def track_history(self):
        """
        Track current hyperparameters in the history.
        """
        History.alpha.append(self.alpha)
        History.beta.append(self.beta)