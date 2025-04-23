import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        decay = self.adapt_decay_based_on_performance()  # Dynamic decay adjustment
        pheromone *= decay  # Evaporate pheromone levels
        normalized_costs = self.normalize_costs(costs)  # Normalize costs for deposition
        n_ants = paths.shape[1]  # Get the number of ants
        
        # Update pheromones based on paths taken and normalized costs
        for i in range(n_ants):
            path = paths[:, i]  # Path taken by the i-th ant
            deposit = normalized_costs[i]  # Get the deposit amount
            pheromone[path[:-1], path[1:]] += deposit  # Update pheromone for edges
            pheromone[path[1:], path[:-1]] += deposit  # Ensure pheromone is symmetric

        self.update_history(decay, pheromone)  # Track history updates
        return pheromone  # Return the updated pheromone tensor

    def adapt_decay_based_on_performance(self) -> float:
        """Adaptive decay based on performance during iterations."""
        if len(History.costs) < 2:
            return 0.9  # Default decay
        current_costs = History.costs[-1]
        previous_costs = History.costs[-2]
        mean_current_cost = current_costs.mean().item()
        mean_previous_cost = previous_costs.mean().item()

        # Adjust decay based on the improvement of mean costs
        if mean_current_cost < mean_previous_cost:
            return max(0.7, 0.9 - (mean_previous_cost - mean_current_cost) / 100)
        return min(0.9, 0.95)  # Increase decay if costs did not improve

    def normalize_costs(self, costs: torch.Tensor) -> torch.Tensor:
        """Normalize costs to a range between 0 and 1 and invert for deposition."""
        costs_min = costs.min()
        costs_range = costs.max() - costs_min + 1e-6  # Avoid division by zero
        normalized = (costs - costs_min) / costs_range  # Normalize to [0, 1]
        return 1 - normalized  # Invert to maximize pheromone on lower costs

    def update_history(self, decay: float, pheromone: torch.Tensor):
        """Updates the history with current decay and pheromone levels."""
        History.decay.append(decay)  # Track decay value
        History.pheromone.append(pheromone.clone())  # Track pheromone levels with clone for immutability
