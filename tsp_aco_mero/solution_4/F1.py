import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values.
    """
    
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to heuristic attractiveness values.
        
        Args:
            distances: Tensor of shape (n, n) with distances between cities
            
        Returns:
            Tensor of shape (n, n) with heuristic values
        """
        # Avoid division by zero
        distances_safe = torch.clamp(distances, min=1e-10)  
        
        # Calculate average distance
        avg_distance = torch.mean(distances_safe)  
        
        # Compute attractiveness based on inverted distances
        exponent = 1.5  
        attractiveness = (1.0 / distances_safe) ** exponent  
        
        # Calculate dynamic penalties based on average distance
        edge_penalty = self.dynamic_penalty(avg_distance, distances) 
        attractiveness -= edge_penalty.view(-1, 1)  
        
        # Ensure non-negative attractiveness values
        attractiveness = torch.clamp(attractiveness, min=0)  
        
        # Record heuristic values for tracking
        History.heuristic.append(attractiveness.clone())
        
        return attractiveness
    
    def dynamic_penalty(self, avg_distance: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """
        Calculate a dynamic penalty based on average distance and distances between cities.
        
        Args:
            avg_distance: The average distance of all city pairs
            distances: The original distance tensor
        
        Returns:
            A tensor representing penalty values for each city
        """
        # Normalize penalty based on average distance
        return 0.1 * (torch.mean(distances, dim=1) - avg_distance)