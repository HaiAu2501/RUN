import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values with enhanced adaptability.
    """

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        distances_safe = distances + 1e-10  # Stability safeguard        
        attractiveness = 1.0 / distances_safe  # Basic heuristic        
        attractiveness = torch.pow(attractiveness, 3)  # Adjusted nonlinear transformation for improved sensitivity        
        attractiveness /= torch.max(attractiveness)  # Normalize for consistent comparison        
        adaptive_heuristic = attractiveness * self.compute_scaling_factor()  # Dynamic scaling        
        History.heuristic.append(adaptive_heuristic)  # Track for later analysis
        return adaptive_heuristic

    def compute_scaling_factor(self) -> torch.Tensor:
        iteration = History.iteration[-1] if History.iteration else 1
        total_iters = History.n_iterations or 1
        scaling_factor = (1 + iteration / total_iters) / 2  # Linear progression based on performance
        return scaling_factor