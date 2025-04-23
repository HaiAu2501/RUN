import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values using a hybrid approach.
    """

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        # Avoid division by zero using a small constant
        distances_safe = distances.clone().clamp(min=1e-10)

        # Normalize distances for robustness
        normalized_distances = distances_safe / torch.max(distances_safe)

        # Compute attractiveness using dynamic weighting based on distances
        dynamic_factors = 1 / (1 + distances_safe)
        h1 = 1 / (normalized_distances + 1e-10)  # Adding epsilon for numerical stability

        # Calculate heuristic values as a product of attractiveness
        attractiveness = h1 * dynamic_factors

        # Normalize attractiveness for better differentiability
        heuristic_values = attractiveness / (torch.max(attractiveness) + 1e-10)  # Prevent division by zero

        # Track heuristic values for analysis
        History.heuristic.append(heuristic_values.clone())

        return heuristic_values