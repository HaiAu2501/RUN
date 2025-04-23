import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """Implementation of pheromone update strategy for ACO TSP with dynamic adaptation and modular components."""
    
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        """Update pheromone levels based on ant paths and solution costs."""
        decay = self.calculate_dynamic_decay()  # Calculate dynamic decay
        pheromone = self.evaporation(pheromone, decay)  # Evaporate pheromone
        contributions = self.calculate_contributions(costs)  # Calculate contributions based on costs
        pheromone = self.deposit_pheromone_contributions(pheromone, paths, contributions)  # Deposit pheromone contributions
        pheromone = self.normalize_pheromone_levels(pheromone)  # Normalize pheromone levels to avoid issues
        History.decay.append(decay)  # Track decay value
        return pheromone
    
    def evaporation(self, pheromone: torch.Tensor, decay: float) -> torch.Tensor:
        """Evaporate pheromone based on a decay factor."""
        return pheromone * decay
    
    def calculate_dynamic_decay(self) -> float:
        """Determine dynamic decay based on historical costs."""
        if len(History.costs) < 2:
            return 0.9  # Default decay if insufficient history
        recent_costs = History.costs[-1]
        previous_costs = History.costs[-2]
        improvement = (previous_costs.mean() - recent_costs.mean()).item()
        return max(0.1, min(0.9, 0.9 - improvement * 0.1))  # Ensure stable decay changes
    
    def calculate_contributions(self, costs: torch.Tensor) -> torch.Tensor:
        """Compute contributions for pheromone deposits based on ant costs."""
        return 1.0 / costs.unsqueeze(1)  # Invert costs to get contributions
    
    def deposit_pheromone_contributions(self, pheromone: torch.Tensor, paths: torch.Tensor, contributions: torch.Tensor) -> torch.Tensor:
        """Deposit pheromone contributions using vectorized operations."""
        n_ants = paths.shape[1]
        for i in range(n_ants):
            path = paths[:, i]
            pheromone[path, path.roll(1)] += contributions[i]  # Use vectorized addition
            pheromone[path.roll(1), path] += contributions[i]  # For undirected edges
        return pheromone
    
    def normalize_pheromone_levels(self, pheromone: torch.Tensor) -> torch.Tensor:
        """Ensure pheromone levels are clamped to prevent numerical issues."""
        return torch.clamp(pheromone, min=1e-10)  # Clamp to avoid zero levels
