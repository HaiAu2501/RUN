import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """Implementation of heuristic strategy for ACO TSP. Converts distances into heuristic attractiveness values with adaptive parameters."""

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # weight for pheromone
        self.beta = beta    # weight for heuristic value

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert distances to heuristic attractiveness values."""
        # Safeguard against division by zero
        distances_safe = self.ensure_numerical_stability(distances)

        # Inverse distances as base heuristic values
        heuristic_values = self.calculate_heuristics(distances_safe)

        # Adjust alpha and beta based on algorithm performance over iterations
        if History.iteration:
            avg_cost = self.calculate_avg_cost(History.costs)
            self.update_weights(avg_cost)
            History.alpha.append(self.alpha)
            History.beta.append(self.beta)

        # Final heuristic values adjusted by alpha and beta
        adjusted_heuristic_values = self.adjust_with_weights(heuristic_values, distances_safe)
        History.heuristic.append(adjusted_heuristic_values.clone())

        return adjusted_heuristic_values

    def ensure_numerical_stability(self, distances):
        return torch.where(distances == 0, torch.tensor(1e-10, device=distances.device), distances)

    def calculate_heuristics(self, distances_safe):
        return (1.0 / distances_safe) / (1.0 / distances_safe).sum(dim=1, keepdim=True)

    def calculate_avg_cost(self, costs):
        return sum(costs) / len(costs) if costs else 1.0

    def update_weights(self, avg_cost):
        self.alpha = max(0.1, self.alpha * (1 - avg_cost))
        self.beta = max(0.1, self.beta * (1 - avg_cost))

    def adjust_with_weights(self, heuristic_values, distances_safe):
        return (heuristic_values ** self.alpha) * (torch.exp(-distances_safe) ** self.beta)