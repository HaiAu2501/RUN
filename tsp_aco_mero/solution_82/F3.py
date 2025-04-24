import torch
from .mero import *


class PheromoneImpl(PheromoneStrategy):
    """Implementation of pheromone update strategy for ACO TSP with dynamic decay and stability measures."""

    def __init__(self, decay_rate=0.1, epsilon=1e-6, max_pheromone=1.0, smoothing_factor=0.9):
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.max_pheromone = max_pheromone
        self.smoothing_factor = smoothing_factor
        self.avg_cost = []

    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
               costs: torch.Tensor) -> torch.Tensor:
        # Update average cost with exponential smoothing
        if self.avg_cost:
            new_avg_cost = (self.smoothing_factor * self.avg_cost[-1] + 
                            (1 - self.smoothing_factor) * costs.mean())
        else:
            new_avg_cost = costs.mean()
        self.avg_cost.append(new_avg_cost)

        # Calculate and apply dynamic decay
        decay = self.calculate_dynamic_decay(new_avg_cost)
        self.apply_evaporation(pheromone, decay)

        # Update pheromones based on ant paths
        self.update_pheromone(pheromone, paths, costs)

        # Log history for analysis
        self.log_history(decay, costs, pheromone)
        return pheromone

    def calculate_dynamic_decay(self, avg_cost: float) -> float:
        trend_factor = min(History.costs[-1]) / avg_cost if History.costs else 1.0
        return max(0.5, 0.9 - 0.1 * trend_factor)

    def apply_evaporation(self, pheromone: torch.Tensor, decay: float):
        pheromone *= decay

    def update_pheromone(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor):
        for i in range(paths.shape[1]):
            path = paths[:, i].long()  # ensure index is long
            deposition_factor = 1.0 / (costs[i] + self.epsilon)
            pheromone[path, torch.roll(path, shifts=1)] += deposition_factor
            pheromone[torch.roll(path, shifts=1), path] += deposition_factor
        pheromone = torch.clamp(pheromone, min=self.epsilon, max=self.max_pheromone)

    def log_history(self, decay: float, costs: torch.Tensor, pheromone: torch.Tensor):
        History.decay.append(decay)
        History.costs.append(costs.clone())
        History.pheromone.append(pheromone.clone())