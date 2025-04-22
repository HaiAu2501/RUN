import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values with improved adaptability and generalization.
    """
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        # Avoid division by zero by replacing zeros in distance with a small value
        distances_safe = distances.clone()
        distances_safe[distances_safe == 0] = 1e-10  
        
        # Compute the rolling mean distance for normalization
        rolling_mean_distance = distances_safe.mean(dim=0)
        normalized_distances = distances_safe / rolling_mean_distance
        
        # Create heuristic values as the inverse of normalized distances
        heuristic_values = 1.0 / normalized_distances
        
        # Track heuristic values for dynamic adjustment
        History.heuristic.append(heuristic_values.clone())
        
        # Adjust heuristic based on historical performance
        dynamic_adjustment = self.adjust_based_on_history(History.heuristic)
        return heuristic_values * dynamic_adjustment
    
    def adjust_based_on_history(self, heuristic_history):
        # Adjust based on historical average values
        if len(heuristic_history) < 2:
            return 1.0  # Default scaling if not enough history
        # Calculate scaling factor based on the change in average heuristic values
        scaling_factor = heuristic_history[-1].mean() / heuristic_history[-2].mean()
        return min(max(scaling_factor, 0.5), 2.0)  # Clamp to ensure stability