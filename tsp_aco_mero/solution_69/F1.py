import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values with dynamic weighting and density considerations.
    """
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        # Avoid division by zero by setting these values to a safe number
        mask = (distances == 0)
        distances_safe = distances.clone()
        distances_safe[mask] = 1e-10
        
        # Dynamic Weight Adjustment
        w1, w2 = self.dynamic_weighting(distances)

        # Calculate density  factors across the distance matrix
        density_factor = self.calculate_density_factors(distances)
        
        # Calculate heuristic attractiveness using vectorized operations
        attractiveness = (w1 * (1.0 / distances_safe) + w2 * density_factor)

        # Normalize the heuristic values
        heuristic = attractiveness / attractiveness.sum(dim=1, keepdim=True)
        
        # Set safe values for zero distances
        heuristic[mask] = 1e10

        return heuristic

    def dynamic_weighting(self, distances):
        """
        Compute dynamic weights based on the distances.
        """
        avg_distance = distances[distances > 0].mean().item()  # Avoid averaging over zero distances
        w1 = 1 / avg_distance
        w2 = avg_distance
        return w1, w2
    
    def calculate_density_factors(self, distances):
        """
        Calculate density factors for heuristic attractiveness.
        This is a placeholder for more contextual evaluations if needed.
        """
        scaling_factor = 1.0 / (distances.size(0) + 1)  # Normalize
        return scaling_factor * torch.ones_like(distances)