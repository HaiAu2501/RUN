import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation with dynamic adaptation.
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
        decay = self.adaptive_decay()
        pheromone = self.apply_evaporation(pheromone, decay)
        pheromone = self.deposit_pheromones(paths, costs, pheromone)
        self.record_history(decay, pheromone)
        return pheromone

    def adaptive_decay(self) -> float:
        """ Returns the dynamically calculated decay rate. """ 
        if len(History.costs) == 0:
            return 0.9  # Default decay
        avg_cost = torch.mean(torch.cat(History.costs)).item()  
        return max(0.5, min(0.95, 0.95 - (avg_cost / 100.0)))

    def apply_evaporation(self, pheromone: torch.Tensor, decay: float) -> torch.Tensor:
        """ Method to apply evaporation. """ 
        return pheromone * decay
 
    def deposit_pheromones(self, paths: torch.Tensor, costs: torch.Tensor, pheromone: torch.Tensor) -> torch.Tensor:
        """ Deposit pheromones based on paths and costs. """ 
        n_ants = paths.shape[1]
        avg_cost = torch.mean(costs).item()
        for i in range(n_ants):
            path = paths[:, i].long()  # Ensure long tensor for indexing
            cost = max(costs[i].item(), 1e-10)  # Avoid division by zero
            deposit_amount = (1.0 / cost) * (avg_cost / cost)
            self.update_pheromone(pheromone, path, deposit_amount)
        return pheromone

    def update_pheromone(self, pheromone: torch.Tensor, path: torch.Tensor, amount: float):
        """ Update pheromone matrix based on path. """ 
        pheromone[path, torch.roll(path, shifts=1)] += amount
        pheromone[torch.roll(path, shifts=1), path] += amount

    def normalize(self, pheromone: torch.Tensor) -> torch.Tensor:
        """ Normalize pheromone levels to maintain stability. """ 
        tau_min, tau_max = 0.001, 10.0  # Limits for normalization
        pheromone = torch.clamp(pheromone, tau_min, tau_max)  # Clamp values
        pheromone = (pheromone - tau_min) / (tau_max - tau_min)  # Smooth normalization
        return pheromone

    def record_history(self, decay: float, pheromone: torch.Tensor):
        """ Track hyperparameters and pheromone state. """ 
        History.decay.append(decay)
        History.pheromone.append(self.normalize(pheromone))