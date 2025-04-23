import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values, incorporating dynamic weighting,
    multi-objective evaluation, normalization, and adaptive learning.
    """

    def compute(self, distances: torch.Tensor, traffic_factor: torch.Tensor = None, 
                time: torch.Tensor = None, terrain: torch.Tensor = None) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        Add optional parameters for dynamic evaluation.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            traffic_factor: Tensor of shape (n, n) with traffic conditions (default None)
            time: Tensor of shape (n, n) with time metrics (default None)
            terrain: Tensor of shape (n, n) with terrain information (default None)
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        # Handling zero distances
        mask = (distances == 0)
        distances_safe = distances.clone()
        distances_safe[mask] = 1e-10  # Prevent division by zero (safety measure)
        
        # Base attractiveness from distances
        attractiveness = 1.0 / distances_safe
        dynamic_factors = []
        
        # Add dynamic factors if available
        if traffic_factor is not None:
            dynamic_factors.append(attractiveness * traffic_factor)
        if time is not None:
            dynamic_factors.append(attractiveness * time)
        if terrain is not None:
            dynamic_factors.append(attractiveness * terrain)
        
        # Normalize the combined dynamic factors if available
        if dynamic_factors:
            combined = sum(dynamic_factors) / len(dynamic_factors)
            combined = combined / (combined.max() if combined.max() > 0 else 1)  # Safe normalization
            History.heuristic.append(combined.clone())
            return combined
        
        # If no dynamic factors, just return the base attractiveness
        History.heuristic.append(attractiveness.clone())
        return attractiveness