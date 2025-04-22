import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation with enhanced strategies.
    """
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
              costs: torch.Tensor) -> torch.Tensor:
        """
        Update pheromone levels based on ant paths and solution costs.
        
        Args:
            pheromone: Tensor of shape (n, n) with current pheromone levels
            paths: Tensor of shape (n_cities, n_ants) with paths taken by ants
            costs: Tensor of shape (n_ants,) with path costs
            
        Returns:
            Updated pheromone tensor
        """
        # Compute dynamic decay based on historical performance
        decay = self.calculate_dynamic_decay()

        # Apply evaporation to the pheromone levels
        pheromone *= decay

        # Identify the elite paths based on the lowest costs
        elite_indices = self.select_elite_paths(costs)

        # Deposit pheromones for elite paths
        self.deposit_pheromones(pheromone, paths, costs, elite_indices)

        # Normalize pheromone levels for stability
        pheromone = self.safe_normalization(pheromone)

        # Log history for decay and pheromone levels
        History.decay.append(decay)
        History.pheromone.append(pheromone.clone())
        
        return pheromone

    def calculate_dynamic_decay(self):
        if History.iteration:
            return max(0.75, min(1.0, 1.0 - History.costs[-1].mean() / 80.0))
        return 0.9

    def select_elite_paths(self, costs):
        elite_count = max(1, int(0.2 * len(costs)))  # Top 20%
        return torch.argsort(costs)[:elite_count]

    def deposit_pheromones(self, pheromone, paths, costs, elite_indices):
        for index in elite_indices:
            path = paths[:, index]
            pheromone[path, torch.roll(path, shifts=-1)] += 1.0 / costs[index]  # Forward edge
            pheromone[torch.roll(path, shifts=-1), path] += 1.0 / costs[index]  # Backward edge

    def safe_normalization(self, pheromone):
        return torch.clamp((pheromone - pheromone.min()) / (pheromone.max() - pheromone.min() + 1e-10) * 9.98 + 0.01, 0.01, 10)