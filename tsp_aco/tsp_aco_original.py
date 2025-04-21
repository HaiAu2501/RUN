import numpy as np
import torch
from typing import List
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import tsplib95
import os

class History:
    """Static class to track key ACO metrics accessible from anywhere."""
    iteration: int = 0

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
        History.costs = []
        History.pheromone = []
        History.heuristic = []
        History.alpha = []
        History.beta = []
        History.decay = []



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


class ProbabilityStrategy(ABC):
    """Strategy interface for computing selection probabilities"""
    
    @abstractmethod
    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities for next city selection.
        
        Args:
            pheromone: Tensor of shape (n, n) with pheromone levels
            heuristic: Tensor of shape (n, n) with heuristic values
            
        Returns:
            Tensor with probability values
        """
        pass


class PheromoneStrategy(ABC):
    """Strategy interface for pheromone deposition and evaporation"""
    
    @abstractmethod
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
        heuristic_strategy: Optional[HeuristicStrategy] = None,
        probability_strategy: Optional[ProbabilityStrategy] = None,
        pheromone_strategy: Optional[PheromoneStrategy] = None
    ) -> None:
        """
        Initialize the ACO solver with configurable strategies.
        """
        # Reset history tracking for new run
        History.reset()
        
        self.device = device
        self.distances = torch.tensor(distances, device=device)
        
        self.n_cities = len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.seed = seed
        
        # Initialize strategies
        self.heuristic_strategy = heuristic_strategy
        self.probability_strategy = probability_strategy
        self.pheromone_strategy = pheromone_strategy
        
        # Check that all strategies are provided
        if not all([heuristic_strategy, probability_strategy, pheromone_strategy]):
            raise ValueError("All strategies must be provided")
        
        # Initialize pheromone matrix
        self.pheromone = torch.ones_like(self.distances, device=device)
        History.pheromone.append(self.pheromone.clone())
        
        # Pre-compute heuristic matrix 
        self.heuristic_matrix = self.heuristic_strategy.compute(self.distances)
        History.heuristic.append(self.heuristic_matrix.clone())
        
        self.best_path = None
        self.best_cost = float('inf')
    
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
            
            probs = self.probability_strategy.compute(
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
        Now without explicit decay parameter (managed by strategy).
        """
        self.pheromone = self.pheromone_strategy.update(
            pheromone=self.pheromone,
            paths=paths,
            costs=costs
        )
        
        History.pheromone.append(self.pheromone.clone())
    
    def update_best_solution(self, costs: torch.Tensor) -> None:
        """
        Update the best solution found so far.
        """
        min_cost, _ = torch.min(costs, dim=0)
        if min_cost < self.best_cost:
            self.best_cost = min_cost.item()
    
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
            self.update_best_solution(costs)
            
            # Update pheromones
            self.update_pheromones(paths, costs)
        
        return self.best_cost

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
        mask = (distances == 0)
        distances_safe = distances.clone()
        distances_safe[mask] = 1e-10  
        
        # Basic inverse distance heuristic
        return 1.0 / distances_safe


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
        beta = 1.0

        probabilities = (pheromone ** alpha) * (heuristic ** beta)

        # Hyperparameter tracking
        History.alpha.append(alpha)
        History.beta.append(beta)
        
        return probabilities


class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Handles pheromone deposition and evaporation.
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
        # Hyperparameter setup
        decay = 0.9

        # Apply evaporation
        pheromone = pheromone * decay
        
        # Deposit new pheromones
        n_ants = paths.shape[1]
        for i in range(n_ants):
            path = paths[:, i]
            cost = costs[i]
            # Add pheromone to edges in the path (both directions for symmetric TSP)
            pheromone[path, torch.roll(path, shifts=1)] += 1.0 / cost
            pheromone[torch.roll(path, shifts=1), path] += 1.0 / cost

        # Hyperparameter tracking
        History.decay.append(decay)
            
        return pheromone

def get_data(name):
	problem = tsplib95.load(f'../benchmark/{name}.tsp')

	nodes = list(problem.get_nodes())
	n = len(nodes)

	distances = np.zeros((n, n))

	for i_idx, i in enumerate(nodes):
		for j_idx, j in enumerate(nodes):
			if i != j:
				distances[i_idx][j_idx] = problem.get_weight(i, j)

	optimal = None
	with open('../solutions', 'r') as f:
		for line in f:
			line = line.strip()
			if not line or ':' not in line:
				continue
			key, val = line.split(':', 1)
			if key.strip() == name:
				optimal = int(val.strip())
	return distances, optimal

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