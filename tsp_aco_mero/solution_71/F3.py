import torch
from .mero import *
import random

class PheromoneImpl(PheromoneStrategy):
    def __init__(self, exploration_threshold=0.1):
        self.exploration_threshold = exploration_threshold

    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
              costs: torch.Tensor) -> torch.Tensor:
        # Calculate dynamic decay
        decay = self.calculate_dynamic_decay(costs)
        pheromone *= decay  # Apply decay
        # Normalize pheromones
        pheromone = self.normalize_pheromones(pheromone)
        self.deposit_stochastic_pheromones(pheromone, paths, costs)
        # Update histories for decay and pheromones
        History.decay.append(decay)
        History.pheromone.append(pheromone.clone())
        return pheromone

    def calculate_dynamic_decay(self, costs):
        avg_cost = costs.mean().item() if costs.numel() > 0 else 1.0
        return max(0.5, min(0.95, 0.9 * (1.0 / avg_cost)))

    def normalize_pheromones(self, pheromone):
        min_pheromone = pheromone.min().item() or 1.0
        return pheromone / min_pheromone

    def deposit_stochastic_pheromones(self, pheromone, paths, costs):
        n_ants = paths.shape[1]
        for i in range(n_ants):
            path = paths[:, i]
            cost = costs[i].item() + 1e-6  # Prevent zero division
            pheromone[path, torch.roll(path, shifts=1)] += (1.0 / cost)
            pheromone[torch.roll(path, shifts=1), path] += (1.0 / cost)
            if random.random() < self.exploration_threshold:
                pheromone[path, torch.roll(path, shifts=1)] += 0.1
                pheromone[torch.roll(path, shifts=1), path] += 0.1
