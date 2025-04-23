import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        # Dynamically adapt alpha and beta based on historical performance
        alpha, beta = self.adapt_dynamic_parameters()

        # Use power scaling to magnify decision influence
        pheromone_scaled = torch.pow(pheromone, 1.5)
        heuristic_scaled = torch.pow(heuristic, 2.5)

        # Calculate attractiveness using dynamic scaling
        attractiveness = (pheromone_scaled ** alpha) * (heuristic_scaled ** beta)

        # Normalize probabilities ensuring numerical stability
        epsilon = 1e-10
        prob_sum = torch.sum(attractiveness, dim=1, keepdim=True) + epsilon
        probabilities = attractiveness / prob_sum

        # Track usage of adaptive parameters
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities

    def adapt_dynamic_parameters(self):
        alpha = 1.0
        beta = 2.0
        if len(History.costs) > 2:
            # Calculate cost improvement to adapt parameters
            cost_improvement = torch.mean(History.costs[-1]) - torch.mean(History.costs[-2])
            if cost_improvement < 0:
                alpha *= 1.05
                beta *= 0.95
            else:
                alpha *= 0.95
                beta *= 1.05
        return alpha, beta
