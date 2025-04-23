import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        # Set initial hyperparameters
        alpha = 1.0   # Importance of pheromone
        beta = 2.0    # Importance of heuristic
        
        # Adjust hyperparameters dynamically based on historical performance
        alpha, beta = self.dynamic_adjustment(alpha, beta)

        # Compute attractiveness values
        attractiveness = self.calculate_attractiveness(pheromone, heuristic, alpha, beta)
        
        # Normalize attractiveness with stable measures
        probabilities = self.robust_normalization(attractiveness)

        # Track hyperparameters in History for analysis
        History.alpha.append(alpha)
        History.beta.append(beta)

        return probabilities

    def dynamic_adjustment(self, alpha: float, beta: float) -> tuple:
        """
        Adjust alpha and beta based on average historical costs.
        Safeguards against aggressive shifts.
        """
        if len(History.costs) > 0:
            previous_cost = History.costs[-1].mean().item()
            alpha = max(0.5, min(1.5, alpha * (0.9 if previous_cost > 10 else 1.1)))
            beta = max(0.5, min(2.5, beta * (1.1 if previous_cost > 10 else 0.9)))
        return alpha, beta

    def calculate_attractiveness(self, pheromone: torch.Tensor, heuristic: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        return (pheromone ** alpha) * (heuristic ** beta)

    def robust_normalization(self, probabilities: torch.Tensor) -> torch.Tensor:
        mask = (probabilities > 0).float()  # Create a mask for valid probabilities
        probabilities *= mask  # Apply mask to probabilities
        norm = probabilities.sum(dim=1, keepdim=True)  # Sum across each row
        norm += 1e-9  # Epsilon to avoid division by zero
        return probabilities / norm  
