import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """
    def __init__(self, alpha=1.0, beta=2.0, decay_rate=0.95, epsilon=1e-10):
        self.alpha = alpha
        self.beta = beta
        self.decay_rate = decay_rate
        self.epsilon = epsilon

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        # Apply pheromone decay
        pheromone = pheromone * self.decay_rate

        # Dynamically adjust hyperparameters based on historical performance
        adaptive_alpha, adaptive_beta = self.auto_adjust_parameters() 

        # Calculate attractiveness based on pheromone and heuristic values
        attractiveness = (pheromone ** adaptive_alpha) * (heuristic ** adaptive_beta)

        # Normalize probabilities robustly to handle numerical issues
        probabilities = self.robust_normalization(attractiveness)

        # Track hyperparameters in History for analysis
        self.track_hyperparameters(adaptive_alpha, adaptive_beta)
        return probabilities

    def auto_adjust_parameters(self):
        # Use broader historical performance to smooth adjustments
        if len(History.costs) > 0:
            previous_cost = torch.mean(torch.stack(History.costs))  # Average over historical costs
            adaptive_alpha = max(self.alpha + (previous_cost.item() * 0.01), 0)  # Ensure non-negative
            adaptive_beta = max(self.beta + (previous_cost.item() * 0.01), 0)  # Ensure non-negative
        else:
            adaptive_alpha = self.alpha
            adaptive_beta = self.beta
        return adaptive_alpha, adaptive_beta

    def robust_normalization(self, attractiveness):
        # Avoid division by zero by adding a small epsilon
        total_attractiveness = torch.sum(attractiveness, dim=1, keepdim=True) + self.epsilon
        normalized_probabilities = attractiveness / total_attractiveness
        return normalized_probabilities

    def track_hyperparameters(self, alpha, beta):
        History.alpha.append(alpha)
        History.beta.append(beta)