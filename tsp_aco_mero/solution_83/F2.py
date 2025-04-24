import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of the probability calculation strategy for ACO in TSP.
    Combines pheromone and heuristic information to calculate transition probabilities.
    """

    def __init__(self, alpha=1.0, beta=1.0, decay=0.1):
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta    # Influence of heuristic
        self.decay = decay  # Pheromone decay rate

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities for next city selection.
        
        Args:
            pheromone: Tensor of shape (n, n) with pheromone levels
            heuristic: Tensor of shape (n, n) with heuristic values
            
        Returns:
            Tensor with probability values normalized for selection
        """
        # Ensure non-negative values to prevent division by zero
        pheromone = torch.clamp(pheromone, min=1e-10)
        heuristic = torch.clamp(heuristic, min=1e-10)

        # Calculate attractiveness of cities
        attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)

        # Normalize probabilities
        total_attractiveness = torch.sum(attractiveness, dim=1, keepdim=True)
        probabilities = torch.where(total_attractiveness == 0, 
                                    torch.zeros_like(attractiveness), 
                                    attractiveness / total_attractiveness)

        # Apply pheromone decay
        pheromone.mul_(1 - self.decay)  # Update pheromone levels

        # Track hyperparameters using History
        History.alpha.append(self.alpha)
        History.beta.append(self.beta)
        History.decay.append(self.decay)

        return probabilities
