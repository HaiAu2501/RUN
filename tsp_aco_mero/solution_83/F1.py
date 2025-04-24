import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values with adaptive scaling.
    """

    def __init__(self, decay_rate: float = 0.5, geographical_bias: torch.Tensor = None):
        self.decay_rate = decay_rate
        self.geographical_bias = geographical_bias

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        # Safe handling of distances to avoid division by zero
        distances_safe = torch.clamp(distances, min=1e-10)
        heuristic_base = 1.0 / distances_safe

        # Calculate dynamic scaling based on mean distances
        mean_distance = torch.mean(distances_safe)
        scale_factor = mean_distance / (distances_safe + 1e-10)
        heuristic_dynamic = scale_factor * heuristic_base

        # Calculate adaptive weight based on historical costs
        adaptive_weight = self.calculate_adaptive_weight()
        heuristic_combined = adaptive_weight * (heuristic_dynamic + heuristic_base) / 2.0

        # Apply geographical bias if provided
        if self.geographical_bias is not None:
            heuristic_combined *= self.geographical_bias

        # Store heuristic values into history
        History.heuristic.append(heuristic_combined.clone())
        return heuristic_combined

    def calculate_adaptive_weight(self) -> float:
        if len(History.costs) > 1:
            return min(max(0.5, History.costs[-1].mean() / History.costs[-2].mean()), 2.0)
        return 1.0