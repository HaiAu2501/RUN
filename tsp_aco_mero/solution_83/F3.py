import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    def __init__(self, alpha=1.0, beta=2.0, base_decay=0.85, sensitivity_parameter=0.1):
        self.alpha = alpha
        self.beta = beta
        self.base_decay = base_decay
        self.sensitivity_parameter = sensitivity_parameter

    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        # Calculate adaptive decay strategy
        decay = self.adaptive_decay(costs)
        pheromone *= decay  # Apply evaporation
        pheromone = torch.clamp(pheromone, min=1e-10)  # Prevent numerical issues

        # Select elite indices based on costs
        elite_indices = self.select_paths(costs)

        # Deposit pheromone based on elite paths
        contributions = self.deposit_pheromone(pheromone, paths, costs, elite_indices)
        pheromone += contributions

        # Track metrics in history
        History.decay.append(decay)
        History.costs.append(costs.clone())
        History.pheromone.append(pheromone.clone())
        return pheromone

    def adaptive_decay(self, costs):
        mean_cost = costs.mean().item()
        decay = max(self.base_decay - self.sensitivity_parameter * (mean_cost / 100), 0.5)
        return min(decay, 0.99)  # Clamp to [0.5, 0.99]

    def select_paths(self, costs):
        return costs.argsort()[:int(0.2 * len(costs))]  # Elite is top 20%

    def deposit_pheromone(self, pheromone, paths, costs, elite_indices):
        contributions = torch.zeros_like(pheromone)
        for i in elite_indices:
            path = paths[:, i]
            cost = costs[i] + 1e-10  # Prevent division by zero
            contribution = self.alpha / cost  # Use alpha for scaling
            contributions[path, torch.roll(path, shifts=1)] += contribution
            contributions[torch.roll(path, shifts=1), path] += contribution
        return contributions
