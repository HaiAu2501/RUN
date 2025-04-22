from tsp_aco_mero.solution_1.mero import *
import numpy as np
import torch
import tsplib95
import os
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod

class HeuristicImpl(HeuristicStrategy):
    """Implementation of heuristic strategy for ACO TSP. Transforms distances into heuristic attractiveness values."""

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert distances to heuristic attractiveness values.""" 
        # 1. Normalize distances to the [0, 1] range
        min_distance = distances.min()
        max_distance = distances.max()
        normalized_distances = (distances - min_distance) / (max_distance - min_distance + 1e-10)

        # 2. Apply a quadratic inverse transformation for attractiveness
        epsilon = 1e-10  # Small constant to avoid division by zero
        attractiveness = 1.0 / (normalized_distances ** 2 + epsilon)

        # 3. Incorporate an adaptive adjustment based on historical performance
        mean_attractiveness = attractiveness.mean(dim=1, keepdim=True)
        adaptive_factor = 1.0 / (1 + (mean_attractiveness / (attractiveness + epsilon)))
        attractiveness *= adaptive_factor

        # 4. Normalize attractiveness values
        attractiveness_normalized = attractiveness / attractiveness.sum(dim=1, keepdim=True)

        # 5. Track heuristic values over iterations
        History.heuristic.append(attractiveness_normalized.clone())
        
        return attractiveness_normalized

epsilon = 1e-10  # Small constant to avoid division by zero

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
            Tensor with normalized probability values
        """
        # Adaptive hyperparameters based on historical performance metrics
        alpha = self.adapt_alpha()  
        beta = self.adapt_beta()  

        # Combine pheromone and heuristic information
        attractiveness = (pheromone ** alpha) * (heuristic ** beta)

        # Normalize attractiveness to get transition probabilities
        probabilities = self.stable_normalization(attractiveness)

        # Track hyperparameters in History for analysis and coordination
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities

    def stable_normalization(self, attractiveness: torch.Tensor) -> torch.Tensor:
        """
        Normalize attractiveness values to convert them into probabilities.
        Avoids issues with division by zero.
        """
        max_attractiveness = torch.max(attractiveness)
        if max_attractiveness > 0:
            normalized_probabilities = (attractiveness / (max_attractiveness + epsilon))
            total_probabilities = torch.sum(normalized_probabilities)
            return normalized_probabilities / (total_probabilities + epsilon)  # Ensure the probabilities sum to 1
        return torch.zeros_like(attractiveness)

    def adapt_alpha(self) -> float:
        """
        Adjust alpha dynamically based on the historical average of the costs to promote stability.
        """  
        if len(History.costs) > 1:
            last_cost = History.costs[-1].mean().item()
            previous_costs = History.costs[-2].mean().item()
            improvement = last_cost - previous_costs
            # Bound alpha adjustments to certain limits to prevent erratic changes
            return max(0.5, min(2.0, 1.0 + improvement / 10.0))  
        return 1.0  # Default value

    def adapt_beta(self) -> float:
        """
        Adjust beta dynamically based on the historical average of the costs to promote stability.
        """    
        if len(History.costs) > 1:
            last_cost = History.costs[-1].mean().item()
            previous_costs = History.costs[-2].mean().item()
            improvement = last_cost - previous_costs
            # Bound beta adjustments similarly to alpha
            return max(0.5, min(2.0, 1.0 + improvement / 10.0))  
        return 1.0  # Default value

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation with dynamic adaptation and
    enhanced stability.
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
        # Dynamic decay calculation based on historical costs and iteration
        decay = self.dynamic_decay()

        # Apply evaporation with numerical stability
        pheromone *= decay

        # Amount to deposit based on costs ensuring stability
        deposit_amount = 1.0 / (costs + 1e-9)

        # Pheromone deposition process with local search
        for i in range(paths.shape[1]):
            path = paths[:, i]  
            improved_path = self.local_search(path)  # Perform local optimization
            self.deposit_pheromone(pheromone, improved_path, deposit_amount[i])

        # Boost the top paths to encourage exploration of high-quality solutions
        self.boost_top_paths(pheromone, paths, costs)

        # Normalize pheromone to avoid overflow
        pheromone /= (torch.max(pheromone) + 1e-9)

        # Tracking history for adaptations
        History.decay.append(decay)
        History.pheromone.append(pheromone.clone())
        
        return pheromone

    def dynamic_decay(self) -> float:
        """
        Calculate a dynamic decay factor based on the best cost observed.
        """  
        base_decay = 0.9
        if History.iteration:
            best_cost = min(History.costs[-1]) if History.costs[-1].size(0) > 0 else 0
            decay = max(0.5, base_decay - (best_cost / 1000))
            return decay
        return base_decay
 
    def deposit_pheromone(self, pheromone: torch.Tensor, path: torch.Tensor, deposit_amount: float):
        """
        Deposits pheromone on the specified path in the pheromone matrix.
        """  
        pheromone[path, torch.roll(path, shifts=1)] += deposit_amount

    def boost_top_paths(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor):
        """
        Boost pheromone levels on top paths to enhance exploration of high-quality solutions.
        """  
        num_top_paths = min(len(costs), 5)  # Boost top 5 paths
        top_indices = torch.topk(costs, num_top_paths, largest=False)[1]
        for idx in top_indices:
            path = paths[:, idx]
            self.deposit_pheromone(pheromone, path, deposit_amount=1.0)

    def local_search(self, path: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for a local search method.
        Should return an improved version of the input path.
        """  
        return path # Placeholder: improve the path in a real implementation.

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
        print(f"Average cost for TSP{size}: {avg_costs}")

if __name__ == "__main__":
    run_aco(n_ants=50, n_iterations=200)