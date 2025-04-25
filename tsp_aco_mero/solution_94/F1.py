import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP using a dynamic game-theoretic model.
    This transforms distances into attractiveness values based on path interactions and historical data.
    """

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to dynamic heuristic attractiveness values based on game theory principles.
        
        Args:
            distances: Tensor of shape (n, n) with distances between nodes
        
        Returns:
            Tensor of shape (n, n) with transformed attractiveness values
        """
        # Safety for numerical stability (avoid division by zero)
        distances_safe = distances.clone().clamp(min=1e-10)

        # Define parameters
        alpha = 0.5  # Weight for historical influence
        beta = 2.0   # Power for path interaction effect

        # Calculate base attractiveness using inverse distance
        base_attractiveness = (1 / distances_safe) ** beta
        base_attractiveness[~base_attractiveness.isfinite()] = 0  # Handle infinities

        # Historical weights coming from previous selections, initializing if empty
        if len(History.heuristic) > 0:
            historical_weights = History.heuristic[-1].detach()
        else:
            historical_weights = torch.ones(*base_attractiveness.shape, device=distances.device)

        # Calculate attractiveness influenced by historical weights
        attractiveness_pre_feedback = base_attractiveness * historical_weights

        # Normalize based on sum of attractiveness
        attractiveness_normalized = attractiveness_pre_feedback / (torch.sum(attractiveness_pre_feedback, dim=1, keepdim=True) + 1e-10)

        # Store heuristic values in history
        History.heuristic.append(attractiveness_normalized.detach())

        return attractiveness_normalized
