import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation with dynamic decay, momentum, and normalization.
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
        # 1. Dynamic decay adjustment based on costs
        decay = self.adjust_decay(costs)
        updated_pheromone = pheromone * decay  # Apply evaporation

        # 2. Calculate new pheromone depositions based on paths
        pheromone_momentum = torch.zeros_like(pheromone)
        for i in range(paths.size(1)):
            path = paths[:, i]
            cost = costs[i].item()  # Get cost as a Python number
            deposit_amount = 1.0 / cost  # Decide how much pheromone to deposit

            # Add pheromone to edges in the path (symmetrical for TSP)
            pheromone_momentum[path, torch.roll(path, shifts=1)] += deposit_amount
            pheromone_momentum[torch.roll(path, shifts=1), path] += deposit_amount

        # 3. Combine current updates with momentum
        updated_pheromone += pheromone_momentum

        # 4. Normalize pheromone levels
        updated_pheromone = self.normalize_pheromone(updated_pheromone)

        # 5. Track hyperparameters in History
        History.decay.append(decay)
        History.pheromone.append(updated_pheromone.clone())  # Storing the updated state

        return updated_pheromone

    def adjust_decay(self, costs: torch.Tensor) -> float:
        """
        Adjust decay rate based on recent performance metrics, with respect to average costs.
        """ 
        average_cost = costs.mean().item()  # Calculate average cost
        if average_cost < 20:  # This threshold can be adjusted based on empirical testing
            return 0.85  # Decay is increased for better performance
        return 0.9  # Standard decay rate

    def normalize_pheromone(self, pheromone: torch.Tensor) -> torch.Tensor:
        """
        Normalize pheromone levels using min-max scaling to maintain stability.
        """ 
        min_pheromone = pheromone.min()
        max_pheromone = pheromone.max()
        if max_pheromone - min_pheromone > 0:
            # Apply min-max normalization
            return (pheromone - min_pheromone) / (max_pheromone - min_pheromone)
        return pheromone  # Return unchanged if all values are the same
