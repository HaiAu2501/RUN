import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    def __init__(self, alpha=1.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        # Avoid division by zero by substituting small values where distances are zero
        mask = (distances == 0)
        distances_safe = distances.clone()
        distances_safe[mask] = 1e-10

        # Compute Inverse Distance Heuristic
        inverse_distances = 1.0 / distances_safe
        heuristic_values = inverse_distances * (inverse_distances ** self.alpha)  # Apply alpha scaling

        
        # Apply Exponential Decay
        decay_values = torch.exp(-distances_safe)  # Apply exponential decay based on the distances
        heuristic_values *= decay_values  # Scale the heuristic values by decay
        
        # Combine with pheromones
        pheromones = History.pheromone[-1] if History.pheromone else torch.ones_like(distances)
        combined_values = heuristic_values * (pheromones ** self.beta)

        # Update history
        History.heuristic.append(combined_values.clone())

        return combined_values