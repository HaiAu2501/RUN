from mero import *
import numpy as np
import torch
import tsplib95
import os
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values.
    """
    
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        # Avoid division by zero
        distances_safe = torch.clamp(distances, min=1e-10)  
        
        # Calculate average distance
        avg_distance = torch.mean(distances_safe)  
        
        # Compute attractiveness based on inverted distances
        exponent = 1.5  
        attractiveness = (1.0 / distances_safe) ** exponent  
        
        # Calculate dynamic penalties based on average distance
        edge_penalty = self.dynamic_penalty(avg_distance, distances) 
        attractiveness -= edge_penalty.view(-1, 1)  
        
        # Ensure non-negative attractiveness values
        attractiveness = torch.clamp(attractiveness, min=0)  
        
        # Record heuristic values for tracking
        History.heuristic.append(attractiveness.clone())
        
        return attractiveness
    
    def dynamic_penalty(self, avg_distance: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """
        Calculate a dynamic penalty based on average distance and distances between cities.
        
        Args:
            avg_distance: The average distance of all city pairs
            distances: The original distance tensor
        
        Returns:
            A tensor representing penalty values for each city
        """
        # Normalize penalty based on average distance
        return 0.1 * (torch.mean(distances, dim=1) - avg_distance)

class ProbabilityImpl(ProbabilityStrategy):
    """Implementation of a probability calculation strategy for ACO TSP."""

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """Compute probabilities for next city selection."""
        alpha, beta = self.adapt_parameters()
        attractiveness = self.calculate_attractiveness(pheromone, heuristic, alpha, beta)
        probabilities = self.normalize(attractiveness)

        # Track hyperparameters in History for analysis and coordination
        History.alpha.append(alpha)
        History.beta.append(beta)
        return probabilities

    def adapt_parameters(self):
        # Set initial parameters
        alpha, beta = 1.0, 2.0  # Default hyperparameters
        if len(History.costs) > 1:
            cost_diff = History.costs[-1].mean() - History.costs[-2].mean()
            alpha, beta = self.fine_tune_params(alpha, beta, cost_diff)
        return alpha, beta

    def fine_tune_params(self, alpha, beta, cost_diff):
        # Implementing a feedback loop for a more dynamic adjustment
        alpha = self.clamp(alpha + self.dynamic_adjustment(cost_diff), 0.1, 5.0)
        beta = self.clamp(beta + self.dynamic_adjustment(cost_diff), 0.1, 5.0)
        return alpha, beta

    def dynamic_adjustment(self, cost_diff):
        return 0.2 * (cost_diff < 0)  # Adjust based on performance improvement or degradation

    def calculate_attractiveness(self, pheromone, heuristic, alpha, beta):
        return (pheromone ** alpha) * (heuristic ** beta)

    def normalize(self, attractiveness):
        total = torch.sum(attractiveness)
        return attractiveness / (total + 1e-6)  # Prevent division by zero

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))


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