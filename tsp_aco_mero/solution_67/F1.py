import torch
from .mero import *

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
        distances_safe = torch.where(distances == 0, 1e-10, distances)

        # Basic inverse distance heuristic
        inverse_distances = 1.0 / distances_safe

        # Normalize to improve differentiation
        max_distance = torch.max(inverse_distances)
        heuristic = inverse_distances / max_distance

        # Dynamically adjusted weights based on the heuristic
        density_weights = self.dynamically_adjusted_weights(heuristic)

        # Track heuristic values in history
        History.heuristic.append(heuristic.cpu())

        return heuristic * density_weights

    def dynamically_adjusted_weights(self, heuristic: torch.Tensor) -> torch.Tensor:
        weights = heuristic / torch.sum(heuristic)  # Normalize based on the sum of heuristics
        return weights / torch.max(weights)  # Ensure weights stay within a valid range
