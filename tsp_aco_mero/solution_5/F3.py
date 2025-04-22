import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """Implementation of pheromone update strategy for ACO TSP.  
    Handles pheromone deposition and evaporation, with adaptive and normalized factors."""

    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
              costs: torch.Tensor) -> torch.Tensor:
        """Update pheromone levels based on ant paths and solution costs."""
        # Fixed decay parameter
        decay = 0.9  
        pheromone *= decay  # Evaporation step
        avg_cost = costs.mean() + 1e-10

        # Adaptive decay calculation
        adaptive_decay = max(0.1, min(0.9, decay * (avg_cost / (costs + 1e-10).mean())))
        pheromone *= adaptive_decay  # Apply adaptive decay

        # Normalized contribution based on costs
        costs_normalized = 1 / (costs + 1e-10)  # Prevent division by zero
        norms = costs_normalized / torch.sum(costs_normalized)  # Normalize costs

        n_ants = paths.shape[1]
        # Update pheromone deposition for all ant paths
        for i in range(n_ants):
            path = paths[:, i]
            deposit_strength = norms[i] * 1.0  # Normalized contribution
            pheromone[path, torch.roll(path, shifts=1)] += deposit_strength
            pheromone[torch.roll(path, shifts=1), path] += deposit_strength

        # Elitist strategy: maintain the best paths dynamically
        best_indices = torch.argsort(costs)[:2]  # Select indices of best paths
        elitism_constant = 5.0
        for idx in best_indices:
            best_path = paths[:, idx]
            pheromone[best_path, torch.roll(best_path, shifts=1)] += elitism_constant / (costs[idx] + 1e-10)  # Reinforce best paths

        # Track update statistics in History
        History.decay.append(adaptive_decay)  # Track decay
        History.pheromone.append(pheromone.clone())  # Track pheromone state
        return pheromone