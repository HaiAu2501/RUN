import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """

    def dynamic_tune(self):
        if len(History.costs) > 1:
            previous_cost = History.costs[-2].mean()
            current_cost = History.costs[-1].mean()
            alpha = min(4.0, 1.0 + (previous_cost - current_cost) / previous_cost)
            beta = min(4.0, 2.0 + (previous_cost - current_cost) / previous_cost)
        else:
            alpha, beta = 1.0, 2.0  # Static fallback values
        return alpha, beta

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.dynamic_tune()  # Use adaptive tuning

        # Compute attractiveness scores using pheromone and heuristic
        attractiveness = (pheromone ** alpha) * (heuristic ** beta)

        # Normalize to obtain probabilities
        probabilities = self.normalize(attractiveness)

        # Track hyperparameters in history for analysis and coordination
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Normalize the values for probabilistic interpretation.
        """
        values = torch.clamp(values, min=1e-10)  # Avoid zero values
        return values / (values.sum() + 1e-10)  # Normalize to ensure valid probabilities