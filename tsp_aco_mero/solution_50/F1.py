import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    def __init__(self, alpha=2.0, beta=1.0, gamma=1.0, random_factor=0.05):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.random_factor = random_factor

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        # Safe distance handling to avoid division by zero
        safe_distances = torch.clamp(distances, min=1e-10)

        # Step 1: Apply power-law scaling transformation
        power_law_result = safe_distances ** -self.alpha

        # Step 2: Apply exponential decay transformation
        exp_decay_result = torch.exp(-self.beta * safe_distances)

        # Step 3: Apply pheromone scaling if pheromone history exists
        if History.pheromone:
            current_pheromone = History.pheromone[-1]
            pheromone_scaled_result = (current_pheromone / safe_distances) ** self.gamma
        else:
            pheromone_scaled_result = torch.zeros_like(safe_distances)

        # Step 4: Implement dynamic weighting based on environmental feedback
        current_iter = History.iteration[-1] if History.iteration else 0
        weights = self.calculate_dynamic_weights(safe_distances, current_iter)

        # Step 5: Combine strategies using dynamic weights
        combined_result = (weights[0] * power_law_result + 
                           weights[1] * exp_decay_result + 
                           weights[2] * pheromone_scaled_result)

        # Step 6: Add adaptive noise
        noise = self.random_factor * torch.rand_like(combined_result) * (1.0 / (1.0 + current_iter))
        heuristic_values = combined_result + noise

        # Step 7: Normalize heuristic values for stability
        heuristic_values = heuristic_values / (heuristic_values.sum(dim=1, keepdim=True) + 1e-10)

        # Step 8: Update the heuristic history
        History.heuristic.append(heuristic_values.clone())
        History.alpha.append(self.alpha)
        History.beta.append(self.beta)

        return heuristic_values

    def calculate_dynamic_weights(self, distances, current_iter):
        # Define arbitrary weighting scheme for illustration
        base_weight = 0.333
        iter_modifier = min(0.01 * current_iter, 0.2)  # Cap to avoid excessive weight distortion
        return [base_weight + iter_modifier, base_weight, base_weight - iter_modifier]
