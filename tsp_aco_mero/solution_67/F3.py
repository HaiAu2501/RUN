import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation with enhancements.
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
        # Hyperparameter setup
        initial_decay_rate = 0.9
        epsilon = 1e-6  # To avoid division by zero
        top_k = 5  # Number of elite solutions to prioritize

        # Calculate best and mean costs
        best_cost = torch.min(costs) 
        mean_cost = torch.mean(costs) 

        # Dynamic decay rate based on performance
        adaptive_decay = initial_decay_rate * (best_cost / (mean_cost + epsilon))

        # Apply evaporation
        pheromone *= adaptive_decay
        
        # Select elite indices based on costs
        elite_indices = costs.argsort()[:top_k]
        
        # Deposit new pheromones for elite solutions
        for i in elite_indices:
            path = paths[:, i]
            deposit_value = (1.0 / (costs[i] + epsilon))  # Avoid division by zero
            pheromone[path, torch.roll(path, shifts=1)] += deposit_value

        # Track hyperparameters for future adjustments
        History.decay.append(adaptive_decay)
        History.alpha.append(1.0)  # Tracking arbitrary value for potential use
        History.beta.append(1.0)  # Assuming standard exploitation weight
        
        return pheromone