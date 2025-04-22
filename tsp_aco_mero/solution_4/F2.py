import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """Implementation of a probability calculation strategy for ACO TSP."""

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """Compute probabilities for next city selection."""
        alpha, beta = self.adapt_parameters()
        attractiveness = self.calculate_attractiveness(pheromone, heuristic, alpha, beta)
        probabilities = self.normalize(attractiveness)

        # Track hyperparameters in History for analysis and coordination
        History.alpha.append(alpha)
        History.beta.append(beta)
        return probabilities

    def adapt_parameters(self):
        # Set initial parameters
        alpha, beta = 1.0, 2.0  # Default hyperparameters
        if len(History.costs) > 1:
            cost_diff = History.costs[-1].mean() - History.costs[-2].mean()
            alpha, beta = self.fine_tune_params(alpha, beta, cost_diff)
        return alpha, beta

    def fine_tune_params(self, alpha, beta, cost_diff):
        # Implementing a feedback loop for a more dynamic adjustment
        alpha = self.clamp(alpha + self.dynamic_adjustment(cost_diff), 0.1, 5.0)
        beta = self.clamp(beta + self.dynamic_adjustment(cost_diff), 0.1, 5.0)
        return alpha, beta

    def dynamic_adjustment(self, cost_diff):
        return 0.2 * (cost_diff < 0)  # Adjust based on performance improvement or degradation

    def calculate_attractiveness(self, pheromone, heuristic, alpha, beta):
        return (pheromone ** alpha) * (heuristic ** beta)

    def normalize(self, attractiveness):
        total = torch.sum(attractiveness)
        return attractiveness / (total + 1e-6)  # Prevent division by zero

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))