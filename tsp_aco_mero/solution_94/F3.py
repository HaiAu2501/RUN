import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
              costs: torch.Tensor) -> torch.Tensor:
        """
        Update pheromone levels based on ant paths and solution costs, utilizing an adaptive memory approach.
        
        Args:
            pheromone: Tensor of shape (n, n) with current pheromone levels
            paths: Tensor of shape (n_cities, n_ants) with paths taken by ants
            costs: Tensor of shape (n_ants,) with path costs
            
        Returns:
            Updated pheromone tensor
        """
        # Decay pheromones with adaptive strategy based on iteration
        decay = 0.9 * (1 - History.iteration / History.n_iterations)  # Dynamic decay rate

        pheromone *= decay  # Apply evaporation

        # Calculate min-max normalization of costs with stabilization using a log transformation
        min_cost = costs.min()
        max_cost = costs.max()
        normalized_costs = (costs - min_cost) / (max_cost - min_cost + 1e-6)  # Normalize costs

        # Apply non-linear transformation to amplify differences
        utilities = torch.log1p(1 / (normalized_costs + 1e-6))  # Log scaling for deeper insights
        utilities /= utilities.sum()  # Normalize utilities

        # Non-linear reinforcement that includes path quality influence
        for i in range(paths.shape[1]):
            path = paths[:, i].long()  # Retrieve path index
            cost_inverse = 1 / (costs[i] + 1e-6)  # Inverse cost to promote better paths
            reinforcement = utilities[i] * cost_inverse  # Calculate the reinforcement value

            # Update pheromones based on paths taken (both directions)
            pheromone[path, torch.roll(path, shifts=1)] += reinforcement
            pheromone[torch.roll(path, shifts=1), path] += reinforcement

        # Track history for performance monitoring and adaptive learning
        History.pheromone.append(pheromone.clone())
        History.decay.append(decay)
        History.alpha.append(None)

        return pheromone
