from mero import *
import numpy as np
import torch
import tsplib95
import os
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod

class HeuristicImpl(HeuristicStrategy):
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon  # small value to prevent division by zero

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        # Safeguard for distance computations
        distances_safe = distances.clone() + self.epsilon
        mean_distance = torch.mean(distances_safe)
        variance_distance = torch.var(distances_safe)
        adaptive_alpha = self.calculate_scaling_factor(mean_distance, variance_distance)
        heuristic_values = (1.0 / distances_safe) ** adaptive_alpha
        
        # Store the computed heuristic values in History for tracking
        History.heuristic.append(heuristic_values.clone())
        return heuristic_values

    def calculate_scaling_factor(self, mean_distance, variance_distance):
        # Calculate adaptive scaling factor based on mean and variance
        return 2.0 / (mean_distance + 0.5 * variance_distance)  # Enhanced stability and adaptability

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
        # Set fixed hyperparameters for stability
        alpha = 1.0  # Importance of pheromone
        beta = 1.0   # Importance of heuristic
        
        # Normalize pheromone and heuristic values
        pheromone_normalized = pheromone + 1e-8  # Prevent division by zero
        heuristic_normalized = heuristic + 1e-8  # Prevent division by zero

        attractiveness = (pheromone_normalized ** alpha) * (heuristic_normalized ** beta)

        # Normalize attractiveness to get probabilities
        probabilities = attractiveness / (attractiveness.sum(dim=1, keepdim=True) + 1e-8)  # Prevent division by zero

        # Track hyperparameters in History for analysis
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities


class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation with adaptive mechanisms.
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
        
        # Adaptive decay based on average costs
        average_cost = costs.mean().item()
        decay = self.calculate_decay(average_cost)  # Dynamic decay rate
        pheromone = self.apply_evaporation(pheromone, decay)
        pheromone = self.weighted_update(pheromone, paths, costs)
        self.elitist_update(pheromone, paths, costs)
        pheromone = self.robust_normalize(pheromone)

        # Store decay and pheromone levels in History
        History.decay.append(decay)
        History.pheromone.append(pheromone.clone())  # Store updated pheromone levels
        
        return pheromone
    
    def calculate_decay(self, average_cost):
        # Adapt decay based on performance
        return max(0.5, 1 - average_cost / (average_cost + 1))

    def apply_evaporation(self, pheromone, decay):
        return torch.clamp(pheromone * decay, min=1e-10)

    def weighted_update(self, pheromone, paths, costs):
        for i in range(paths.shape[1]):
            weight = 1.0 / (costs[i].item() ** 2 + 1e-10)  # Inverse cost to promote better paths
            pheromone[paths[:, i], torch.roll(paths[:, i], shifts=1)] += weight
        return pheromone

    def elitist_update(self, pheromone, paths, costs):
        best_idx = costs.argmin()  # Get index of the best path
        best_path = paths[:, best_idx]  # Get the best path taken
        pheromone[best_path, torch.roll(best_path, shifts=1)] += 1.0  # Boost pheromone for the best path
        return pheromone

    def robust_normalize(self, pheromone):
        return pheromone / (pheromone.sum(dim=1, keepdim=True) + 1e-10)  # Prevent saturation of pheromone values


def run_aco(n_ants=30, n_iterations=100):
    # Lấy tất cả các file trong thư mục benchmark
    for size in [20, 50, 100]:
        avg_costs = 0
        for i in range(1, 65):
            path = f"tsp_aco_mero/test/TSP{size}_{i:02}.npy"
            distances = np.load(path)
            aco = AntColonyOptimization(
                distances=distances,
                n_ants=n_ants,
                n_iterations=n_iterations,
                seed=0,
                pheromone_strategy=PheromoneImpl(),
                heuristic_strategy=HeuristicImpl(),
                probability_strategy=ProbabilityImpl()
            )
            cost = aco.run()
            avg_costs += cost
        avg_costs /= 64
        print(f"Average cost for TSP{size}: {avg_costs:.2f}")

if __name__ == "__main__":
    run_aco(n_ants=30, n_iterations=100)