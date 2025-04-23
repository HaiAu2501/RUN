import torch
from .mero import *

epsilon = 1e-8  # Small constant to prevent division by zero

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation with dynamic adjustments.
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
        # Dynamic decay adjustment based on average cost
        average_cost = torch.mean(costs)
        decay = max(0.5, min(0.9, 1.0 - average_cost / torch.max(costs + epsilon)))  # adaptive decay

        # Evaporation phase
        pheromone = pheromone * decay

        # Robust normalization to prevent numerical issues
        pheromone_sum = pheromone.sum(dim=1, keepdim=True) + epsilon  # Avoid division by zero
        pheromone = pheromone / pheromone_sum

        # Deposit new pheromones based on ant paths
        for i in range(paths.shape[1]):
            path = paths[:, i].long()  # Ensure indices are long
            cost = costs[i].item()     # Get the cost as standard float
            if cost > 0:
                pheromone[path[:-1], path[1:]] += 1.0 / cost  # Deposit on the edges in the path
                pheromone[path[1:], path[:-1]] += 1.0 / cost  # Symmetric deposit

        # Track decay and pheromone levels in history
        History.decay.append(decay)
        History.pheromone.append(pheromone.clone())

        return pheromone
