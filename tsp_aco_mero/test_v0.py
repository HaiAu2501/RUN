import importlib 
import numpy as np
import torch
import tsplib95
import os
import sys
from typing import List, Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm

###########

class History:
    """Static class to track key ACO metrics accessible from anywhere."""
    iteration: int = 0
    n_ants: int = 0
    n_iterations: int = 0

    costs: List[torch.Tensor] = []
    pheromone: List[torch.Tensor] = []
    heuristic: List[torch.Tensor] = []

    # Hyperparameters
    alpha: List[float] = []
    beta: List[float] = []
    decay: List[float] = []

    @staticmethod
    def reset():
        """Reset all history metrics."""
        History.iteration = 0
        History.n_ants = 0
        History.costs = []
        History.pheromone = []
        History.heuristic = []
        History.alpha = []
        History.beta = []
        History.decay = []

###########

# SOLUTION 77

class HeuristicStrategy(ABC):
    """Strategy interface for computing heuristic values from distances"""
    
    @abstractmethod
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        pass


class AntColonyOptimization:
    """
    Optimized implementation of Ant Colony Optimization for TSP.
    Only Heuristic strategy is configurable, other strategies are implemented directly.
    """
    
    def __init__(
        self,
        distances: np.ndarray,
        n_ants: int = 50,
        n_iterations: int = 100,
        device: str = 'cpu',
        seed: int = 123,
        heuristic_strategy: Optional[HeuristicStrategy] = None
    ) -> None:
        """
        Initialize the ACO solver with only a configurable heuristic strategy.
        """
        # Reset history tracking for new run
        History.reset()
        History.n_ants = n_ants
        History.n_iterations = n_iterations
        
        self.device = device
        self.distances = torch.tensor(distances, device=device)
        
        self.n_cities = len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.seed = seed
        
        # Initialize the heuristic strategy
        self.heuristic_strategy = heuristic_strategy
        
        # Check that heuristic strategy is provided
        if not heuristic_strategy:
            raise ValueError("Heuristic strategy must be provided")
        
        # Initialize pheromone matrix
        self.pheromone = torch.ones_like(self.distances, device=device)
        History.pheromone.append(self.pheromone.clone())
        
        # Pre-compute heuristic matrix using the strategy
        self.heuristic_matrix = self.heuristic_strategy.compute(self.distances)
        History.heuristic.append(self.heuristic_matrix.clone())
        
        self.best_path = None
        self.best_cost = float('inf')
    
    def compute_probabilities(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
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
        beta = 1.0 

        probabilities = (pheromone ** alpha) * (heuristic ** beta)

        # Hyperparameter tracking
        History.alpha.append(alpha)
        History.beta.append(beta)
        
        return probabilities
    
    def construct_solutions(self) -> torch.Tensor:
        """
        Optimized path construction for all ants.
        """
        n_cities = self.n_cities
        n_ants = self.n_ants
        device = self.device
        
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        
        # Choose random starting cities
        start = torch.randint(low=0, high=n_cities, size=(n_ants,), device=device)
        
        # Initialize mask for visited cities
        mask = torch.ones(size=(n_ants, n_cities), device=device)
        mask[torch.arange(n_ants, device=device), start] = 0
        
        # Initialize paths with starting cities
        paths_list = [start]
        current_cities = start
        
        # Pre-allocate tensors for improved performance
        arange_ants = torch.arange(n_ants, device=device)
        
        # Construct the rest of the paths
        for _ in range(n_cities - 1):
            pheromone_values = self.pheromone[current_cities]
            heuristic_values = self.heuristic_matrix[current_cities]
            
            # Compute probabilities using the internal method
            probs = self.compute_probabilities(
                pheromone=pheromone_values,
                heuristic=heuristic_values
            )
            
            probs = probs * mask
            
            # Normalize probabilities with optimized approach
            row_sums = torch.sum(probs, dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            probs = probs / row_sums
            
            next_cities = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            paths_list.append(next_cities)
            
            # Update current cities and mask in-place where possible
            current_cities = next_cities
            mask[arange_ants, next_cities] = 0
            
        return torch.stack(paths_list)
    
    def calculate_path_costs(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Optimized calculation of path costs.
        """
        paths_t = paths.permute(1, 0)
        next_cities = torch.roll(paths_t, shifts=1, dims=1)
        costs = torch.sum(self.distances[paths_t, next_cities], dim=1)
        return costs
    
    def update_pheromones(self, paths: torch.Tensor, costs: torch.Tensor) -> None:
        """
        Update pheromone levels based on ant paths - vectorized where possible.
        """
        # Hyperparameter setup
        decay = 0.9

        # Apply evaporation
        self.pheromone = self.pheromone * decay
        
        # Deposit new pheromones
        n_ants = paths.shape[1]
        for i in range(n_ants):
            path = paths[:, i]
            cost = costs[i]
            # Add pheromone to edges in the path (both directions for symmetric TSP)
            self.pheromone[path, torch.roll(path, shifts=1)] += 1.0 / cost
            self.pheromone[torch.roll(path, shifts=1), path] += 1.0 / cost

        # Hyperparameter tracking
        History.decay.append(decay)
        History.pheromone.append(self.pheromone.clone())
    
    def update_best_solution(self, paths: torch.Tensor, costs: torch.Tensor) -> None:
        """
        Update the best solution found so far.
        """
        min_cost, min_idx = torch.min(costs, dim=0)
        if min_cost < self.best_cost:
            self.best_cost = min_cost.item()
            self.best_path = paths[:, min_idx].clone()
    
    def run(self) -> float:
        """
        Run the ACO algorithm with optimized memory usage.
        """        
        for iteration in range(self.n_iterations):
            History.iteration = iteration
            
            self.seed += 1
            
            # Construct solutions for all ants
            paths = self.construct_solutions()
            
            # Calculate path costs
            costs = self.calculate_path_costs(paths)
            
            History.costs.append(costs.clone())
            
            # Update best solution
            self.update_best_solution(paths, costs)
            
            # Update pheromones
            self.update_pheromones(paths, costs)
        
        return self.best_cost


class HeuristicImpl(HeuristicStrategy):
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        self.prevent_div_zero_and_normalize(distances)
        geographical_heuristic = self.calculate_geographical_heuristic(distances)
        performance_metric = self.compute_performance_metric()
        heuristic_values = self.adaptive_thresholding(performance_metric, distances, geographical_heuristic)
        self.adjust_for_exploration(heuristic_values)
        self.record_history(heuristic_values)
        return heuristic_values

    def prevent_div_zero_and_normalize(self, distances):
        mask = (distances == 0)
        distances[mask] = 1e-10  # Avoiding division by zero
        min_val = distances.min()
        max_val = distances.max()
        distances.sub_(min_val).div_(max_val - min_val)  # Normalize in-place

    def calculate_geographical_heuristic(self, distances):
        return 1.0 / (distances + 1.0)  # Return heuristic inversely proportional to distance

    def compute_performance_metric(self):
        if History.costs:
            return History.costs[-1].mean().item()
        return 0.0  # Default when there is no historical data

    def adaptive_thresholding(self, performance_metric, distances, geographical_heuristic):
        threshold = 0.5  # Sample threshold for adaptability
        if performance_metric < threshold:
            return 1.0 / (distances + 1e-10)
        return geographical_heuristic  # Use geographical heuristic otherwise

    def adjust_for_exploration(self, heuristic_values):
        perturbation_factor = 0.1
        perturbation = perturbation_factor * (torch.rand_like(heuristic_values) - 0.5)
        heuristic_values.add_(perturbation).clamp_(min=0)  # Add perturbation and ensure non-negativity

    def record_history(self, heuristic_values):
        History.heuristic.append(heuristic_values.clone())  # Clone to avoid mutability issues

###########

size = int(sys.argv[1])
n_ants = int(sys.argv[2])
n_iterations = int(sys.argv[3])

# def run_aco(size):
#     # Lấy tất cả các file trong thư mục benchmark
#     avg_costs = 0
#     for i in range(1, 65):
#         path = f"tsp_aco_mero/test/TSP{size}_{i:02}.npy"
#         distances = np.load(path)
#         aco = AntColonyOptimization(
#             distances=distances,
#             n_ants=n_ants,
#             n_iterations=n_iterations,
#             seed=0,
#             heuristic_strategy=HeuristicImpl()
#         )
#         cost = aco.run()
#         avg_costs += cost
#     avg_costs = avg_costs / 64
#     print(f"MERO - Average cost for TSP{size}: {avg_costs}")

def run_aco(size):
    avg_costs = 0
    path = f"tsp_aco_mero/ls_tsp/TSP{size}.npy"
    prob_batch = np.load(path)
    from scipy.spatial import distance_matrix
    # Calculate the distance matrix
    for i, prob in enumerate(prob_batch):
        print(f"Processing TSP{size} {i}")
        distances = distance_matrix(prob, prob)
        aco = AntColonyOptimization(
            distances=distances,
            n_ants=n_ants,
            n_iterations=n_iterations,
            seed=0,
            heuristic_strategy=HeuristicImpl()
        )
        cost = aco.run()
        print(f"Cost for TSP{size} {i}: {cost}")
        avg_costs += cost
    avg_costs /= len(prob_batch)
    print(f"Average cost for TSP{size}: {avg_costs}")

if __name__ == "__main__":
    run_aco(size)