import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        distances_safe = self.safe_division(distances)  # Safely handle zero distances
        heuristic_values = self.non_linear_transformation(distances_safe)  # Apply non-linear transformation
        scaling_factor = self.dynamic_scaling(distances)  # Compute dynamic scaling
        heuristic_values *= scaling_factor  # Scale the heuristic values
        History.heuristic.append(heuristic_values.clone())  # Track the heuristics
        return heuristic_values

    def safe_division(self, distances):
        mask = (distances == 0)
        distances_safe = distances.clone()
        distances_safe[mask] = 1e-10  # Replace zero distances with a small value
        return distances_safe

    def non_linear_transformation(self, distances_safe):
        return torch.pow(1.0 / distances_safe, 2)  # Non-linear transformation for attractiveness

    def dynamic_scaling(self, distances):
        mean_distance = distances[distances != 0].mean()  # Mean distance ignoring zeros
        density = distances.sum(dim=1) / distances.size(1)  # Compute average density
        adjusted_density = 1.0 / (1.0 + density.unsqueeze(1))  # Adjust density for scaling
        scale_factor = 1.0 / (mean_distance + 1e-10) * adjusted_density  # Compute scaling factor
        return scale_factor