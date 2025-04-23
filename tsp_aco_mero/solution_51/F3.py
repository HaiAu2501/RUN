import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """Implementation of pheromone update strategy for ACO in TSP. Handles pheromone deposition and evaporation."""

    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
               costs: torch.Tensor) -> torch.Tensor:
        """Update pheromone levels based on ant paths and solution costs."""
        # Calculate average cost across all ants
        avg_cost = torch.mean(costs)
        max_cost = torch.max(costs) + 1e-6  # Prevent division by zero
        
        # Compute adaptive decay value, clamped between 0.8 and 0.95
        decay = self.adaptive_decay(avg_cost, max_cost)
        self.apply_evaporation(pheromone, decay)
        self.deposit_pheromones(pheromone, paths, costs)
        self.normalize_pheromone(pheromone)
        self.track_history(decay, costs, pheromone)
        return pheromone
    
    def adaptive_decay(self, avg_cost, max_cost):
        return max(0.8, min(0.95, 0.9 - avg_cost / max_cost))
    
    def apply_evaporation(self, pheromone, decay):
        pheromone *= decay
    
    def deposit_pheromones(self, pheromone, paths, costs):
        n_ants = paths.shape[1]
        for i in range(n_ants):
            path = paths[:, i]
            cost = costs[i] + 1e-6  # Prevent division by zero in cost calculations
            pheromone[path, torch.roll(path, shifts=1)] += 1.0 / cost
            pheromone[torch.roll(path, shifts=1), path] += 1.0 / cost
        
    def normalize_pheromone(self, pheromone):
        pheromone = torch.clamp(pheromone, min=1e-6)  # Prevent values too low
        if pheromone.sum() > 0:
            pheromone /= pheromone.sum()  # Normalize to sum to 1
        else:
            pheromone = torch.ones_like(pheromone) / pheromone.numel()  # Ensure valid distribution
    
    def track_history(self, decay, costs, pheromone):
        History.decay.append(decay)
        History.costs.append(costs.clone())
        History.pheromone.append(pheromone.clone())
