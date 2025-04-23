import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    Calculates probabilities for selecting the next city based on pheromone and heuristic values.
    """

    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        # Hyperparameter setup with defaults
        alpha, beta = self.get_hyperparameters()  # Retrieve hyperparameters from history
        epsilon = 1e-10  # Small value to avoid division by zero

        # Apply clamping for numerical stability
        pheromone = torch.clamp(pheromone, min=epsilon)  # safeguard against zero
        heuristic = torch.clamp(heuristic, min=epsilon)  # safeguard against zero

        # Calculate attractiveness while ensuring numerical stability
        attractiveness = (pheromone ** alpha) * (heuristic ** beta)
        probabilities = attractiveness / (torch.sum(attractiveness, dim=1, keepdim=True) + epsilon)

        # Update hyperparameters in History for analysis and coordination
        History.alpha.append(alpha)
        History.beta.append(beta)

        # Dynamic adaptation of hyperparameters based on historical progress
        self.update_hyperparameters()  # Update based on performance history

        return probabilities

    def get_hyperparameters(self):
        # Get most recent hyperparameters from history or default values
        return (History.alpha[-1] if History.alpha else 1.0, History.beta[-1] if History.beta else 2.0)

    def update_hyperparameters(self):
        # Adjust alpha and beta based on average cost history
        if len(History.costs) > 0:
            current_cost = History.costs[-1].mean().item()  # recent average cost

            # Gradual adjustment logic for alpha and beta based on previous performance
            new_alpha = max(0.1, 1.0 + 0.01 * (2 - current_cost))  # sensitivity to cost reductions
            new_beta = max(0.1, 2.0 - 0.01 * current_cost)  # sensitivity to cost increases
            History.alpha.append(new_alpha)
            History.beta.append(new_beta)