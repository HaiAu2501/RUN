import torch
import random
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation with adaptive features.
    """
    def __init__(self, base_decay=0.9, min_dec=0.5, max_dec=0.95, stochasticity=0.1):
        self.base_decay = base_decay  # Initial decay rate
        self.min_dec = min_dec  # Minimum decay
        self.max_dec = max_dec  # Maximum decay
        self.stochasticity = stochasticity  # Stochastic update factor

    def adaptive_decay(self):
        """Adjusts decay based on historical performance"""
        if len(History.costs) > 1:
            recent_cost = History.costs[-1].mean()
            previous_cost = History.costs[-2].mean()
            if recent_cost < previous_cost:
                return max(self.min_dec, self.base_decay - 0.1)  # Reduce decay if improvement
            else:
                return min(self.max_dec, self.base_decay + 0.1)  # Increase decay if no improvement
        return self.base_decay

    def clamp_values(self, pheromone):
        """Clamp pheromone values to avoid numerical instability"""
        return torch.clamp(pheromone, min=1e-10, max=1e10)

    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        # Select dynamic decay
        decay = self.adaptive_decay()  
        History.decay.append(decay)
        
        # Apply evaporation
        pheromone *= decay
        
        # Compute average cost and variance
        avg_cost = costs.mean() 
        cost_variance = costs.var()  
        
        # Deposit new pheromones with improved strategy based on variance
        n_ants = paths.shape[1]
        for i in range(n_ants):
            path = paths[:, i]
            pheromone_increment = 1.0 / (avg_cost + cost_variance + 1e-10)  # Adjust based on average and variance
            
            # Stochastic updates with controlled exploration rate
            if random.random() < self.stochasticity:
                adjustment = random.uniform(0.5, 1.0) * (1 + cost_variance)  # Scale adjustment by variance
                pheromone[path, torch.roll(path, shifts=1)] += pheromone_increment * adjustment
                pheromone[torch.roll(path, shifts=1), path] += pheromone_increment * adjustment
            else:
                pheromone[path, torch.roll(path, shifts=1)] += pheromone_increment
                pheromone[torch.roll(path, shifts=1), path] += pheromone_increment
        
        # Normalize and clamp pheromone values
        pheromone = self.clamp_values(pheromone)

        # Update History with the new pheromone values
        History.pheromone.append(pheromone.clone())

        return pheromone
