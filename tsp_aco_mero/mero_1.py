from mero import *
import numpy as np
import torch
import tsplib95
import os
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod

class HeuristicImpl(HeuristicStrategy):
    def __init__(self, alpha=1.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        # Avoid division by zero by substituting small values where distances are zero
        mask = (distances == 0)
        distances_safe = distances.clone()
        distances_safe[mask] = 1e-10

        # Compute Inverse Distance Heuristic
        inverse_distances = 1.0 / distances_safe
        heuristic_values = inverse_distances * (inverse_distances ** self.alpha)  # Apply alpha scaling

        
        # Apply Exponential Decay
        decay_values = torch.exp(-distances_safe)  # Apply exponential decay based on the distances
        heuristic_values *= decay_values  # Scale the heuristic values by decay
        
        # Combine with pheromones
        pheromones = History.pheromone[-1] if History.pheromone else torch.ones_like(distances)
        combined_values = heuristic_values * (pheromones ** self.beta)

        # Update history
        History.heuristic.append(combined_values.clone())

        return combined_values

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities for next city selection.
        
        Args:
            pheromone: Tensor of shape (n, n) with pheromone levels
            heuristic: Tensor of shape (n, n) with heuristic values
            
        Returns:
            Tensor with probability values
        """
        # Hyperparameter setup
        alpha = 1.0
        beta = 2.0
        decay_factor = 0.95  # Dynamic decay factor for pheromone levels

        # Apply dynamic decay to pheromone levels
        pheromone = pheromone * decay_factor

        # Sum of attractiveness values
        probabilities = (pheromone ** alpha) * (heuristic ** beta)

        # Handle numerical stability by normalizing the probabilities
        probabilities_sum = probabilities.sum(dim=1, keepdim=True)
        probabilities = probabilities / probabilities_sum

        # Hyperparameter tracking
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities

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




def run_aco(n_ants=30, n_iterations=100):
    # Lấy tất cả các file trong thư mục benchmark
    for file in os.listdir('benchmark'):
        if file.endswith('.tsp'):
            # Lấy tên file mà không có phần mở rộng
            name = os.path.splitext(file)[0]
            # Đọc dữ liệu từ file
            distances, optimal = get_data(name)
            def opt_gap(optimal, obj):
                return (obj - optimal) / optimal * 100

            # Chạy hàm solve_reevo với dữ liệu đã đọc
            avg_obj = 0
            for seed in range(5):
                heuristic_strategy = HeuristicImpl()
                probability_strategy = ProbabilityImpl()
                pheromone_strategy = PheromoneImpl()
                
                aco = AntColonyOptimization(
                    distances=distances,
                    n_ants=n_ants,
                    n_iterations=n_iterations,
                    device='cpu',
                    seed=seed,
                    heuristic_strategy=heuristic_strategy,
                    probability_strategy=probability_strategy,
                    pheromone_strategy=pheromone_strategy
                )
                
                obj = aco.run()
                avg_obj += obj

            avg_obj /= 5
            print(f"{name}: opt_gap = {opt_gap(optimal, avg_obj):.2f}%")

if __name__ == "__main__":
    run_aco(n_ants=30, n_iterations=100)