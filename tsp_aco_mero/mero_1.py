from mero import *
import numpy as np
import torch
import tsplib95
import os
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod

class HeuristicImpl(HeuristicStrategy):
    """Implementation of heuristic strategy for ACO TSP. Transforms distances into heuristic attractiveness values."""

    def __init__(self, alpha=0.5, exponent=2):
        self.alpha = alpha  # Weight for the hybrid heuristic (inverse distance vs. normalized)
        self.exponent = exponent  # Exponent for polynomial transformation of inverse distances

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert distances to heuristic attractiveness values."""
        # Handle edge cases to prevent division by zero
        distances_safe = distances.clone() + 1e-10  # Add a small value to avoid division by zero

        # Compute inverse distance heuristic
        inverse_heuristic = 1.0 / distances_safe

        # Normalize the distances to range [0, 1]
        normalized_heuristic = self.normalize(distances)

        # Hybrid heuristic combining inverse distance with normalized distances
        composite_heuristic = self.alpha * inverse_heuristic + (1 - self.alpha) * normalized_heuristic

        # Apply polynomial transformation to inverse distance for better sensitivity (optional)
        polynomial_heuristic = (1.0 / distances_safe) ** self.exponent

        # Combine polynomial heuristic with the composite heuristic
        composite_heuristic = self.alpha * polynomial_heuristic + (1 - self.alpha) * normalized_heuristic

        # Track heuristics in history to enable adaptive mechanisms
        History.heuristic.append(composite_heuristic.clone())

        return composite_heuristic

    def normalize(self, distances: torch.Tensor) -> torch.Tensor:
        """Normalize the distance matrix to range [0, 1]."""
        min_dist = torch.min(distances)
        max_dist = torch.max(distances)
        normalized_values = (distances - min_dist) / (max_dist - min_dist)
        return normalized_values

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """

    def dynamic_tune(self):
        if len(History.costs) > 1:
            previous_cost = History.costs[-2].mean()
            current_cost = History.costs[-1].mean()
            alpha = min(4.0, 1.0 + (previous_cost - current_cost) / previous_cost)
            beta = min(4.0, 2.0 + (previous_cost - current_cost) / previous_cost)
        else:
            alpha, beta = 1.0, 2.0  # Static fallback values
        return alpha, beta

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.dynamic_tune()  # Use adaptive tuning

        # Compute attractiveness scores using pheromone and heuristic
        attractiveness = (pheromone ** alpha) * (heuristic ** beta)

        # Normalize to obtain probabilities
        probabilities = self.normalize(attractiveness)

        # Track hyperparameters in history for analysis and coordination
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Normalize the values for probabilistic interpretation.
        """
        values = torch.clamp(values, min=1e-10)  # Avoid zero values
        return values / (values.sum() + 1e-10)  # Normalize to ensure valid probabilities

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation with enhanced strategies.
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
        # Compute dynamic decay based on historical performance
        decay = self.calculate_dynamic_decay()

        # Apply evaporation to the pheromone levels
        pheromone *= decay

        # Identify the elite paths based on the lowest costs
        elite_indices = self.select_elite_paths(costs)

        # Deposit pheromones for elite paths
        self.deposit_pheromones(pheromone, paths, costs, elite_indices)

        # Normalize pheromone levels for stability
        pheromone = self.safe_normalization(pheromone)

        # Log history for decay and pheromone levels
        History.decay.append(decay)
        History.pheromone.append(pheromone.clone())
        
        return pheromone

    def calculate_dynamic_decay(self):
        if History.iteration:
            return max(0.75, min(1.0, 1.0 - History.costs[-1].mean() / 80.0))
        return 0.9

    def select_elite_paths(self, costs):
        elite_count = max(1, int(0.2 * len(costs)))  # Top 20%
        return torch.argsort(costs)[:elite_count]

    def deposit_pheromones(self, pheromone, paths, costs, elite_indices):
        for index in elite_indices:
            path = paths[:, index]
            pheromone[path, torch.roll(path, shifts=-1)] += 1.0 / costs[index]  # Forward edge
            pheromone[torch.roll(path, shifts=-1), path] += 1.0 / costs[index]  # Backward edge

    def safe_normalization(self, pheromone):
        return torch.clamp((pheromone - pheromone.min()) / (pheromone.max() - pheromone.min() + 1e-10) * 9.98 + 0.01, 0.01, 10)


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