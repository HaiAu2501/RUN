import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """Implementation of heuristic strategy for ACO TSP. Transforms distances into heuristic attractiveness values."""

    def __init__(self, alpha=0.5, exponent=2):
        self.alpha = alpha  # Weight for the hybrid heuristic (inverse distance vs. normalized)
        self.exponent = exponent  # Exponent for polynomial transformation of inverse distances

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert distances to heuristic attractiveness values."""
        # Handle edge cases to prevent division by zero
        distances_safe = distances.clone() + 1e-10  # Add a small value to avoid division by zero

        # Compute inverse distance heuristic
        inverse_heuristic = 1.0 / distances_safe

        # Normalize the distances to range [0, 1]
        normalized_heuristic = self.normalize(distances)

        # Hybrid heuristic combining inverse distance with normalized distances
        composite_heuristic = self.alpha * inverse_heuristic + (1 - self.alpha) * normalized_heuristic

        # Apply polynomial transformation to inverse distance for better sensitivity (optional)
        polynomial_heuristic = (1.0 / distances_safe) ** self.exponent

        # Combine polynomial heuristic with the composite heuristic
        composite_heuristic = self.alpha * polynomial_heuristic + (1 - self.alpha) * normalized_heuristic

        # Track heuristics in history to enable adaptive mechanisms
        History.heuristic.append(composite_heuristic.clone())

        return composite_heuristic

    def normalize(self, distances: torch.Tensor) -> torch.Tensor:
        """Normalize the distance matrix to range [0, 1]."""
        min_dist = torch.min(distances)
        max_dist = torch.max(distances)
        normalized_values = (distances - min_dist) / (max_dist - min_dist)
        return normalized_values