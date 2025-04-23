import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        
        # Dynamic adaptation: adapting hyperparameters over iterations
        current_iter = History.iteration
        max_iter = 1000  # assume a typical maximum number of iterations
        alpha = 1.0 + (current_iter / max_iter)  # scale alpha up linearly
        beta = 2.0 + (max_iter - current_iter) / max_iter  # scale beta down

        # Calculate base probabilities using pheromone and heuristic information
        base_probabilities = (pheromone ** alpha) * (heuristic ** beta)
        
        # Normalization to ensure probabilities are within scale
        row_sums = base_probabilities.sum(dim=1, keepdim=True)
        probabilities = base_probabilities / row_sums

        # Track hyperparameters in History
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities