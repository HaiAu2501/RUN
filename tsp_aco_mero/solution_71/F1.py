import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Converts distances into heuristic attractiveness values with adaptive adjustments.
    """

    def __init__(self, multiplicative_factor=1.0):
        self.multiplicative_factor = multiplicative_factor

    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values, with necessary safeguards.
        
        Args:
            distances: Tensor of shape (n, n) representing distances between cities.
            
        Returns:
            Tensor of shape (n, n) containing heuristic values.
        """  
        # Avoiding division by zero and preparing distances
        distances_safe = self.prepare_distances(distances)
        
        # Calculate the base heuristic using the inverse distance approach
        base_heuristic = self.calculate_inverse_heu(distances_safe)
        
        # Use a dynamic adjustment based on historical data and performance metrics
        adaptive_factor = self.calculate_dynamic_scaling()
        
        # Final heuristic values
        heuristic = base_heuristic * adaptive_factor * self.multiplicative_factor 
        History.heuristic.append(heuristic)
        return heuristic

    def prepare_distances(self, distances):
        """
        Prepares distances to avoid division by zero by replacing 0 with a small epsilon value.
        """ 
        distances_safe = distances.clone()  
        distances_safe[distances == 0] = 1e-10  
        return distances_safe

    def calculate_inverse_heu(self, distances_safe):
        """
        Calculates the heuristic values as the inverse of distances ensuring proper scaling.
        """  
        return 1.0 / distances_safe

    def calculate_dynamic_scaling(self) -> float:
        """
        Generates an adaptive scaling factor based on historical performance metrics to adjust heuristic values.
        """  
        adaptive_factor = 1.0  # Default neutral multiplier
        if History.costs:
            # Use recent cost information to adaptively adjust heuristic attractiveness
            recent_cost = History.costs[-1].mean()  
            if recent_cost > 1e-10:
                adaptive_factor += 1.0 / recent_cost
        return adaptive_factor