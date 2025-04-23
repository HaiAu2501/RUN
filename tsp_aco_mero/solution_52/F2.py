import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """Implementation of probability calculation strategy for ACO TSP. Calculates probabilities for selecting the next city based on pheromone and heuristic values."""

    def __init__(self):
        self.previous_alpha = 1.0
        self.previous_beta = 2.0

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        # Dynamic adjustment of hyperparameters with momentum
        alpha = self.dynamic_adjust_alpha_with_momentum()
        beta = self.dynamic_adjust_beta_with_momentum()

        # Normalize heuristic and clamp pheromone for numerical stability
        heuristic_normalized = heuristic / (heuristic.sum(dim=1, keepdim=True) + 1e-10)
        pheromone_clamped = torch.clamp(pheromone, min=1e-10)

        # Calculate attractiveness values
        attractiveness = (pheromone_clamped ** alpha) * (heuristic_normalized ** beta)
        # Normalize to probabilities
        probabilities = attractiveness / (attractiveness.sum(dim=1, keepdim=True) + 1e-10)

        # Track the hyperparameters for analysis and adjustments
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities

    def dynamic_adjust_alpha_with_momentum(self):
        # Implementing momentum in alpha adjustments
        momentum = 0.9
        if len(History.costs) > 0:
            previous_cost = History.costs[-1].mean().item()
            new_alpha = momentum * self.previous_alpha + (1 - momentum) * min(max(previous_cost / 10.0, 1.0), 5.0)
            self.previous_alpha = new_alpha  # Update previous alpha
        else:
            new_alpha = 1.0
        return new_alpha

    def dynamic_adjust_beta_with_momentum(self):
        # Implementing additional logic for beta adjustment using momentum
        momentum = 0.9
        if len(History.costs) > 0:
            recent_cost = History.costs[-1].mean().item()
            new_beta = 3.0 if recent_cost < 20 else 1.5
            new_beta = momentum * self.previous_beta + (1 - momentum) * new_beta
            self.previous_beta = new_beta  # Update previous beta
        else:
            new_beta = 2.0
        return new_beta