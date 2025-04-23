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

class InitializationStrategy(ABC):
    """Strategy interface for initializing pheromone and heuristic matrices"""
    
    @abstractmethod
    def initialize(self, distances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize pheromone and heuristic matrices.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tuple of (pheromone, heuristic) tensors
        """
        pass


class ConstructionStrategy(ABC):
    """Strategy interface for constructing ant paths"""
    
    @abstractmethod
    def construct(self, distances: torch.Tensor, probs: torch.Tensor, 
                  n_ants: int, seed: int) -> torch.Tensor:
        """
        Construct paths for all ants.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            probs: Tensor of shape (n, n) with pre-calculated probabilities
                   (computed from pheromone and heuristic, already masked and normalized)
            n_ants: Number of ants
            seed: Random seed for reproducibility
            
        Returns:
            Tensor of shape (n_cities, n_ants) with paths for all ants
        """
        pass


class UpdateStrategy(ABC):
    """Strategy interface for updating pheromone levels"""
    
    @abstractmethod
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
               costs: torch.Tensor) -> torch.Tensor:
        """
        Update pheromone levels.
        
        Args:
            pheromone: Tensor of shape (n, n) with current pheromone levels
            paths: Tensor of shape (n_cities, n_ants) with paths for all ants
            costs: Tensor of shape (n_ants,) with path costs
            
        Returns:
            Updated pheromone tensor
        """
        pass


class AntColonyOptimization:
    """
    Optimized implementation of Ant Colony Optimization for TSP.
    """
    
    def __init__(
        self,
        distances: np.ndarray,
        n_ants: int = 50,
        n_iterations: int = 100,
        device: str = 'cpu',
        seed: int = 123,
        initialization_strategy: Optional[InitializationStrategy] = None,
        construction_strategy: Optional[ConstructionStrategy] = None,
        update_strategy: Optional[UpdateStrategy] = None
    ) -> None:
        """
        Initialize the ACO solver with configurable strategies.
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
        
        # Initialize strategies
        self.initialization_strategy = initialization_strategy
        self.construction_strategy = construction_strategy
        self.update_strategy = update_strategy
        
        # Check that all strategies are provided
        if not all([initialization_strategy, construction_strategy, update_strategy]):
            raise ValueError("All strategies must be provided")
        
        # Initialize using initialization strategy
        self.pheromone, self.heuristic_matrix = self.initialization_strategy.initialize(self.distances)
        
        self.best_cost = float('inf')
    
    def run(self) -> float:
        """
        Run the ACO algorithm with the provided strategies.
        """  
        torch.manual_seed(self.seed)

        for iteration in range(self.n_iterations):
            History.iteration = iteration
            
            # Calculate probabilities using pheromone and heuristic
            alpha = 1.0
            beta = 1.0
            History.alpha.append(alpha)
            History.beta.append(beta)
            
            # Calculate base probabilities from pheromone and heuristic
            probs = (self.pheromone ** alpha) * (self.heuristic_matrix ** beta)            
            
            # Construct solutions using construction strategy
            paths = self.construction_strategy.construct(
                distances=self.distances,
                probs=probs,
                n_ants=self.n_ants
            )
            
            # Calculate path costs
            paths_t = paths.permute(1, 0)  # Shape: (n_ants, n_cities)
            next_cities = torch.roll(paths_t, shifts=1, dims=1)
            costs = torch.sum(self.distances[paths_t, next_cities], dim=1)
            
            # Store costs in history
            History.costs.append(costs.clone())
            
            # Find best solution
            min_cost, _ = torch.min(costs, dim=0)
            min_cost = min_cost.item()
            
            # Update best overall solution
            if min_cost < self.best_cost:
                self.best_cost = min_cost
            
            # Update pheromones using update strategy
            self.pheromone = self.update_strategy.update(
                pheromone=self.pheromone,
                paths=paths,
                costs=costs
            )
        
        return self.best_cost


###########

# SOLUTION 72

class InitializationImpl(InitializationStrategy):
    """
    Implementation of initialization strategy for ACO TSP.
    Initializes pheromone and heuristic matrices with dynamic and adaptive strategies.
    """
    
    def initialize(self, distances: torch.Tensor, alpha: float = 1.0, beta: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize pheromone and heuristic matrices based on city distances.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            alpha: Weight parameter for pheromone influence
            beta: Weight parameter for heuristic influence
            
        Returns:
            Tuple of (pheromone, heuristic) tensors
        """
        # Ensure distances is a float tensor
        distances = distances.float()  
        
        # Prevent division by zero for heuristics  
        mask = (distances == 0)
        distances_safe = distances.clone()
        distances_safe[mask] = 1e-10  # Avoid division by zero
        
        # Initialize pheromone matrix with a constant value
        pheromone = torch.ones_like(distances) * 1.0
        
        # Calculate heuristic values as the inverse of the distances, scaled by beta
        heuristic = (1.0 / distances_safe) ** beta
        
        # Normalize pheromone and heuristic matrices
        pheromone = pheromone / pheromone.sum(dim=1, keepdim=True)
        heuristic = heuristic / heuristic.sum(dim=1, keepdim=True)
        
        # Track history for adaptive parameter adjustments
        History.pheromone.append(pheromone.clone())
        History.heuristic.append(heuristic.clone())
        History.alpha.append(alpha)
        History.beta.append(beta)
        
        return pheromone, heuristic

class ConstructionImpl(ConstructionStrategy):
    """
    Implementation of solution construction strategy for ACO TSP.
    Constructs paths for all ants based on pre-calculated probabilities.
    """
    def __init__(self, noise_injection_prob=0.1, alpha=1.0, beta=2.0):
        self.noise_injection_prob = noise_injection_prob
        self.alpha = alpha
        self.beta = beta

    def construct(self, distances: torch.Tensor, probs: torch.Tensor, n_ants: int) -> torch.Tensor:
        n_cities = distances.shape[0]
        start = torch.randint(low=0, high=n_cities, size=(n_ants,)).to(distances.device)

        # Initialize mask for visited cities
        mask = torch.ones(size=(n_ants, n_cities), dtype=torch.float32, device=distances.device)
        mask[torch.arange(n_ants), start] = 0
        paths_list = [start]
        current_cities = start.clone()  # Track current cities of each ant

        arange_ants = torch.arange(n_ants, device=distances.device)

        for _ in range(n_cities - 1):
            current_probs = probs[current_cities]
            masked_probs = current_probs * mask

            # Normalize probabilities
            row_sums = torch.sum(masked_probs, dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            normalized_probs = masked_probs / row_sums

            # Noise injection for exploration
            if torch.rand(1).item() < self.noise_injection_prob:
                # Randomly select from unvisited cities
                next_cities = torch.randint(low=0, high=n_cities, size=(n_ants,), device=distances.device)
                valid_choices = mask[arange_ants, next_cities] == 1
                next_cities = torch.where(valid_choices, next_cities, current_cities)
            else:
                next_cities = torch.multinomial(normalized_probs, num_samples=1).squeeze(1)

            paths_list.append(next_cities)
            current_cities = next_cities
            mask[arange_ants, next_cities] = 0

        final_paths = torch.stack(paths_list)
        path_costs = self.calculate_costs(final_paths, distances)
        History.costs.append(path_costs)

        self.adjust_parameters(final_paths)  # Dynamically adjust hyperparameters
        return final_paths

    def calculate_costs(self, paths: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """Calculate the costs of all paths given the distances matrix."""
        total_costs = torch.zeros(paths.shape[1], device=distances.device)
        n_cities = distances.shape[0]
        for ant in range(paths.shape[1]):
            path = paths[:, ant]
            cost = distances[path[:-1], path[1:]].sum() + distances[path[-1], path[0]]
            total_costs[ant] = cost
        return total_costs

    def adjust_parameters(self, final_paths):
        """Adjust hyperparameters based on performance metrics"""
        # Implement feedback mechanism for dynamic tuning here
        # Example: Adjust noise_injection_prob based on the variability of path costs
        if len(History.costs) > 1:
            last_costs = History.costs[-1]
            previous_costs = History.costs[-2]
            performance_change = last_costs.mean() - previous_costs.mean()
            if performance_change < 0:
                self.noise_injection_prob = min(1.0, self.noise_injection_prob + 0.05)
            else:
                self.noise_injection_prob = max(0.0, self.noise_injection_prob - 0.05)

import random

class UpdateImpl(UpdateStrategy):
    def __init__(self, base_decay=0.9, noise_base=0.1, min_decay=0.1):
        self.base_decay = base_decay  # Base decay factor
        self.noise_base = noise_base  # Base noise value for randomness
        self.min_decay = min_decay      # Minimum decay threshold

    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        # Calculate dynamic decay based on path costs
        dynamic_decay = self.calculate_dynamic_decay(costs)
        pheromone *= dynamic_decay  # Evaporate existing pheromone levels

        n_ants = paths.shape[1]  # Number of ants
        pheromone_delta = torch.zeros_like(pheromone)  # Prepare for new pheromone deposits

        for i in range(n_ants):
            path = paths[:, i]  # Retrieve the path for the i-th ant
            deposit = self.calculate_deposit(costs[i])  # Get deposit for current ant

            # Update pheromone for the path (deposited on both sides)
            pheromone_delta[path[:-1], path[1:]] += deposit
            pheromone_delta[path[1:], path[:-1]] += deposit

        pheromone += pheromone_delta  # Apply the accumulated deposits

        # Record current pheromone levels and decay in history for analysis
        History.decay.append(dynamic_decay)
        History.pheromone.append(pheromone.clone())
        return pheromone

    def calculate_dynamic_decay(self, costs):
        average_cost = costs.mean().item()  # Calculate mean cost
        # Dynamically adjust decay based on average path cost
        dynamic_decay = self.base_decay * (average_cost / (torch.max(costs).item() + 1e-6))
        return max(dynamic_decay, self.min_decay)  # Ensure decay does not go below min_decay

    def calculate_deposit(self, cost):
        noise_val = self.noise_base * random.uniform(-1, 1)  # Random noise addition
        # Return deposit calculated inversely to cost with noise adjustment
        deposit = max(0, (1 / (cost + 1e-6)) + noise_val)  # Avoid division by zero
        return deposit


###########



size = sys.argv[1]

def run_aco(size):
    avg_costs = 0
    if size == "200":
        n_ants = 100
        n_iterations = 200
    elif size == "500":
        n_ants = 100
        n_iterations = 200
    elif size == "1000":
        n_ants = 100
        n_iterations = 200
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
            initialization_strategy=InitializationImpl(),
            construction_strategy=ConstructionImpl(),
            update_strategy=UpdateImpl(),
        )
        cost = aco.run()
        print(f"Cost for TSP{size} {i}: {cost}")
        avg_costs += cost
    avg_costs /= len(prob_batch)
    print(f"Average cost for TSP{size}: {avg_costs}")

if __name__ == "__main__":
    run_aco(size)