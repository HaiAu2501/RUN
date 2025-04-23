import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """Implementation of heuristic strategy for ACO TSP with dynamic adjustments for adaptability and stability."""
    def __init__(self, alpha_initial=1.0, epsilon=1e-10, smooth_factor=0.1):
        self.alpha = alpha_initial  # Power law coefficient
        self.epsilon = epsilon  # Small value to prevent division by zero
        self.smooth_factor = smooth_factor  # Controls responsiveness of alpha adjustments

    def update_alpha(self, distances: torch.Tensor):
        """Adjust alpha based on the variability of distance measurements."""
        variation = torch.mean(torch.abs(distances - torch.mean(distances)))
        self.alpha += self.smooth_factor * variation
        self.alpha = max(0.1, min(5.0, self.alpha))  # Clamping alpha to reasonable bounds

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert distances to heuristic attractiveness values, logging the calculations."""
        self.update_alpha(distances)  # Update alpha before computing heuristics
        distances_safe = distances.clone()
        distances_safe[distances_safe == 0] = self.epsilon  # Prevent division by zero

        heuristic_values = distances_safe.pow(-self.alpha)  # Apply inverse power

        # Normalize heuristic values to maintain stability
        heuristic_values /= (heuristic_values.sum(dim=1, keepdim=True) + self.epsilon)  # Normalize with epsilon to maintain stability

        # Store heuristic values for analysis
        History.heuristic.append(heuristic_values.clone())
        return heuristic_values
