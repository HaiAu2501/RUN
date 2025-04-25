import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    """
    Implementation of probability calculation strategy for ACO TSP.
    This version innovatively combines pheromone and heuristic signals while balancing exploration and exploitation decisions using dynamic learning rates.
    """
    
    def compute(self, pheromone: torch.Tensor, heuristic: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities for next city selection using adaptive strategies.
        
        Args:
            pheromone: Tensor of shape (n, n) with pheromone levels
            heuristic: Tensor of shape (n, n) with heuristic values
            
        Returns:
            Tensor with normalized probability values
        """
        # Hyperparameter setup
        alpha = 1.5  # Importance of pheromone
        beta = 1.5   # Importance of heuristic
        decay_rate = 0.1  # Pheromone decay rate

        # Logarithmic transformation for numerical stability
        heuristic_transformed = torch.log1p(heuristic)  

        # Performance metrics to dynamically modulate alpha and beta
        if len(History.costs) > 2:  
            recent_costs = torch.mean(torch.stack(History.costs[-3:]))  # Last three iterations
            performance_gain = (torch.mean(History.costs[-3]) - recent_costs) / torch.mean(History.costs[-3])  
            alpha = 1.0 + performance_gain * 0.5  # Adjust exploration-exploitation balance
            beta = max(0.1, 1.5 - performance_gain * 0.5)  # Scale beta inversely to performance gain

        # Update pheromone levels according to the decay function
        pheromone_updated = pheromone * (1 - decay_rate)

        # Calculate combined scores using a bilinear form to emphasize interactions
        combined_scores = (pheromone_updated ** alpha) * (heuristic_transformed ** beta)

        # Normalize combined scores to ensure they add up to 1 across rows
        row_sums = combined_scores.sum(dim=1, keepdim=True)
        probabilities = combined_scores / (row_sums + 1e-6)  # Preventing division by zero

        # Track hyperparameters in history
        History.alpha.append(alpha)
        History.beta.append(beta)
        History.decay.append(decay_rate)  # Track decay rate

        return probabilities
