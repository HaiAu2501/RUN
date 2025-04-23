import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        # Dynamically adapt parameters based on historical performance
        alpha, beta = self.adapt_parameters()

        # Compute attractiveness based on pheromone and heuristic values
        attractiveness = (pheromone ** alpha) * (heuristic ** beta)

        # Normalize attractiveness to get transition probabilities
        probabilities = self.robust_normalization(attractiveness)

        # Track historical parameters for analysis
        History.alpha.append(alpha)
        History.beta.append(beta)
        return probabilities

    def adapt_parameters(self) -> (float, float):
        recent_cost = History.costs[-1].mean().item() if History.costs else 100.0
        return self.adapt_alpha(recent_cost), self.adapt_beta(recent_cost)

    def adapt_alpha(self, recent_cost) -> float:
        if recent_cost < 80:
            return min(2.0, 1.5 + (1.0 / (recent_cost + 1e-10)))
        elif recent_cost > 120:
            return max(0.5, 1.0 - (1.0 / (recent_cost + 1e-10)))
        return 1.0

    def adapt_beta(self, recent_cost) -> float:
        if recent_cost < 80:
            return min(2.0, 1.5 + (1.0 / (recent_cost + 1e-10)))
        elif recent_cost > 120:
            return max(0.5, 1.0 - (1.0 / (recent_cost + 1e-10)))
        return 1.0

    def robust_normalization(self, attract: torch.Tensor) -> torch.Tensor:
        total = attract.sum(dim=1, keepdim=True)
        # Ensure robust handling of edge cases with normalization
        return attract / total if (total.sum() != 0 and total.max() != 0) else torch.full_like(attract, 1.0 / attract.size(1))
