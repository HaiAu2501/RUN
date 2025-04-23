import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """
    def __init__(self, alpha=1.0, beta=1.0, decay_factor=0.95, default=0.1):
        self.alpha = alpha
        self.beta = beta
        self.decay_factor = decay_factor
        self.default = default

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        # Adapt hyperparameters using historical data
        self.alpha, self.beta = self.adapt_hyperparameters()

        # Calculate attractiveness
        attractiveness = self.calculate_attractiveness(pheromone, heuristic)

        # Normalize the probabilities with stability measures
        probabilities = self.normalize_probabilities(attractiveness)

        # Track hyperparameter changes in history
        History.alpha.append(self.alpha)
        History.beta.append(self.beta)

        return probabilities

    def adapt_hyperparameters(self):
        adaptive_alpha = self.dynamically_adjust(self.alpha, History.alpha)
        adaptive_beta = self.dynamically_adjust(self.beta, History.beta)
        return adaptive_alpha, adaptive_beta

    def dynamically_adjust(self, value, history):
        decay = self.decay_factor
        default = self.default
        return (history[-1] * decay + default) if history else default

    def calculate_attractiveness(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        return (pheromone ** self.alpha) * (heuristic ** self.beta)

    def normalize_probabilities(self, attractiveness: torch.Tensor) -> torch.Tensor:
        sum_attractiveness = torch.sum(attractiveness, dim=1, keepdim=True)
        epsilon = 1e-10  # Stability factor to prevent division by zero
        return attractiveness / (sum_attractiveness + epsilon)