import numpy as np
import torch
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod
import tsplib95

# def get_data(name):
# 	problem = tsplib95.load(f'benchmark/{name}.tsp')

# 	nodes = list(problem.get_nodes())
# 	n = len(nodes)

# 	distances = np.zeros((n, n))

# 	for i_idx, i in enumerate(nodes):
# 		for j_idx, j in enumerate(nodes):
# 			if i != j:
# 				distances[i_idx][j_idx] = problem.get_weight(i, j)

# 	optimal = None
# 	with open('solutions', 'r') as f:
# 		for line in f:
# 			line = line.strip()
# 			if not line or ':' not in line:
# 				continue
# 			key, val = line.split(':', 1)
# 			if key.strip() == name:
# 				optimal = int(val.strip())
# 	return distances, optimal

class History:
    """Static class to track key ACO metrics accessible from anywhere."""
    iteration: int = 0
    n_ants: int = 0
    n_iterations: int = 0

    costs: List[np.ndarray] = []
    pheromone: List[np.ndarray] = []
    heuristic: List[np.ndarray] = []

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

class HeuristicStrategy(ABC):
    """Strategy interface for computing heuristic values from distances"""
    
    @abstractmethod
    def compute(self, distances: np.ndarray) -> np.ndarray:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Array of shape (n, n) with distances between cities
            
        Returns:
            Array of shape (n, n) with heuristic values
        """
        pass


class ProbabilityStrategy(ABC):
    """Strategy interface for computing selection probabilities"""
    
    @abstractmethod
    def compute(self, pheromone: np.ndarray, heuristic: np.ndarray) -> np.ndarray:
        """
        Compute probabilities for next city selection.
        
        Args:
            pheromone: Array of shape (n, n) with pheromone levels
            heuristic: Array of shape (n, n) with heuristic values
            
        Returns:
            Array with probability values
        """
        pass


class PheromoneStrategy(ABC):
    """Strategy interface for pheromone deposition and evaporation"""
    
    @abstractmethod
    def update(self, pheromone: np.ndarray, paths: np.ndarray, 
              costs: np.ndarray) -> np.ndarray:
        """
        Update pheromone levels based on ant paths and solution costs.
        
        Args:
            pheromone: Array of shape (n, n) with current pheromone levels
            paths: Array of shape (n_cities, n_ants) with paths taken by ants
            costs: Array of shape (n_ants,) with path costs
            
        Returns:
            Updated pheromone array
        """
        pass


class AntColonyOptimization:
    """
    Optimized implementation of Ant Colony Optimization for TSP using NumPy.
    """
    
    def __init__(
        self,
        distances: np.ndarray,
        n_ants: int = 50,
        n_iterations: int = 100,
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
        History.n_ants = n_ants
        History.n_iterations = n_iterations
        
        self.distances = distances
        
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
        self.pheromone = np.ones_like(self.distances)
        History.pheromone.append(self.pheromone.copy())
        
        # Pre-compute heuristic matrix 
        self.heuristic_matrix = self.heuristic_strategy.compute(self.distances.copy())
        History.heuristic.append(self.heuristic_matrix.copy())
        
        self.best_path = None
        self.best_cost = float('inf')
    
    def construct_solutions(self) -> np.ndarray:
        """
        Optimized path construction for all ants.
        """
        n_cities = self.n_cities
        n_ants = self.n_ants
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Choose random starting cities
        start = np.random.randint(0, n_cities, size=n_ants)
        
        # Initialize mask for visited cities (1 = unvisited, 0 = visited)
        mask = np.ones((n_ants, n_cities))
        # Mark starting cities as visited
        mask[np.arange(n_ants), start] = 0
        
        # Initialize paths with starting cities
        paths_list = [start]
        current_cities = start
        
        # Construct the rest of the paths
        for _ in range(n_cities - 1):
            # Get pheromone and heuristic values for current cities
            pheromone_values = np.array([self.pheromone[city] for city in current_cities])
            heuristic_values = np.array([self.heuristic_matrix[city] for city in current_cities])
            
            # Compute probabilities using the strategy
            probs = self.probability_strategy.compute(
                pheromone=pheromone_values,
                heuristic=heuristic_values
            )
            
            # Apply mask to exclude visited cities
            probs = probs * mask
            
            # Normalize probabilities to ensure they sum to 1 for each ant
            row_sums = np.sum(probs, axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1.0
            probs = probs / row_sums
            
            # Select next cities using weighted random choice
            next_cities = np.zeros(n_ants, dtype=int)
            for i in range(n_ants):
                # Only choose from cities with non-zero probability
                if np.sum(probs[i]) > 0:
                    next_cities[i] = np.random.choice(n_cities, p=probs[i])
                else:
                    # If all probabilities are zero, choose randomly from unvisited cities
                    unvisited = np.where(mask[i] > 0)[0]
                    if len(unvisited) > 0:
                        next_cities[i] = np.random.choice(unvisited)
                    else:
                        next_cities[i] = np.random.choice(n_cities)
            
            paths_list.append(next_cities)
            
            # Update current cities and mask
            current_cities = next_cities
            mask[np.arange(n_ants), next_cities] = 0
            
        return np.stack(paths_list)
    
    def calculate_path_costs(self, paths: np.ndarray) -> np.ndarray:
        """
        Optimized calculation of path costs.
        """
        # Transpose to get shape (n_ants, n_cities)
        paths_t = paths.transpose(1, 0)
        costs = np.zeros(self.n_ants)
        
        for i in range(self.n_ants):
            # Get path for ant i
            path = paths_t[i]
            # Shift path to get next cities (creates a circular path)
            next_cities = np.roll(path, -1)
            
            # Calculate cost by summing distances between cities
            # Note: This replaces the tensor indexing with standard array indexing
            path_cost = 0
            for j in range(len(path)):
                path_cost += self.distances[path[j], next_cities[j]]
            
            costs[i] = path_cost
            
        return costs
    
    def update_pheromones(self, paths: np.ndarray, costs: np.ndarray) -> None:
        """
        Update pheromone levels based on ant paths.
        """
        self.pheromone = self.pheromone_strategy.update(
            pheromone=self.pheromone,
            paths=paths,
            costs=costs
        )
        
        History.pheromone.append(self.pheromone.copy())
    
    def update_best_solution(self, costs: np.ndarray) -> None:
        """
        Update the best solution found so far.
        """
        min_cost = np.min(costs)
        min_idx = np.argmin(costs)
        
        if min_cost < self.best_cost:
            self.best_cost = min_cost
            # We could also store the best path if needed
            # self.best_path = paths[:, min_idx].copy()
    
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
            
            # Store costs in history
            History.costs.append(costs.copy())
            
            # Update best solution
            self.update_best_solution(costs)
            
            # Update pheromones
            self.update_pheromones(paths, costs)
        
        return self.best_cost