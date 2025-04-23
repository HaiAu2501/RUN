import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of the pheromone update strategy for ACO TSP, enhancing dynamic adaptation and numerical stability.
    """

    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        """
        Update pheromone levels based on ant paths and solution costs. 
        
        Args:
            pheromone: Tensor of shape (n, n) with current pheromone levels
            paths: Tensor of shape (n_cities, n_ants) with paths taken by ants
            costs: Tensor of shape (n_ants,) with path costs
            
        Returns:
            Updated pheromone tensor
        """
        # Dynamically adjust decay rate
        decay = self._adjust_decay()

        # Apply evaporation with clamping for numerical stability
        pheromone = self._clamp(pheromone * decay)
        n_ants = paths.shape[1]

        for i in range(n_ants):
            path = paths[:, i]  # Path taken by ant i
            cost = costs[i].item()  # Cost of the path taken by ant i
            deposit_amount = self._calculate_deposit(cost, costs)

            # Update pheromone for the edges based on the ant's path
            self._deposit(pheromone, path, deposit_amount)

        # Track hyperparameters
        History.decay.append(decay)
        History.pheromone.append(pheromone.clone())
        return pheromone

    def _calculate_deposit(self, cost: float, costs: torch.Tensor) -> float:
        """
        Calculate the amount of pheromone to deposit based on cost relative to others.
        
        Args:
            cost: Cost of the current ant's path.
            costs: Tensor of costs for all ants.
        
        Returns:
            Calculated deposit amount based on relative performance.
        """
        average_cost = torch.mean(costs)
        relative_performance = average_cost / cost if cost > 0 else 0
        return (1.0 / cost) * relative_performance if cost > 0 else 0

    def _adjust_decay(self) -> float:
        """
        Adjust the pheromone decay rate dynamically based on historical performance.
        
        Returns:
            Adjusted decay rate.
        """
        if len(History.costs) < 10:
            return 0.9  # Initial decay value
        recent_costs_mean = torch.mean(torch.stack(History.costs[-10:]))
        return 0.9 if recent_costs_mean > 100 else 0.8  # Adaptive decay based on performance

    def _clamp(self, pheromone: torch.Tensor) -> torch.Tensor:
        """
        Clamp pheromone values to prevent numerical instability.
        
        Args:
            pheromone: Tensor of pheromone levels.
        """
        return torch.clamp(pheromone, min=1e-5, max=100.0)

    def _deposit(self, pheromone: torch.Tensor, path: torch.Tensor, amount: float) -> None:
        """
        Deposit pheromone along the edges defined by the path. 
        
        Args:
            pheromone: Tensor of pheromone levels.
            path: Tensor representing the path taken by the ant.
            amount: Amount of pheromone to deposit.
        """
        pheromone[path, torch.roll(path, shifts=1)] += amount * 0.5
        pheromone[torch.roll(path, shifts=1), path] += amount * 0.5