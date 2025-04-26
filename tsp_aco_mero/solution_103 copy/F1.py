import numpy as np
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """Implementation of heuristic strategy for ACO TSP.
    Transforms distances into heuristic attractiveness values, incorporating adaptive features.
    """

    def compute(self, distances: np.ndarray) -> np.ndarray:
        """Convert distances to heuristic attractiveness values, integrating dynamic factors.
        
        Args:
            distances: Array of shape (n, n) with distances between cities
            
        Returns:
            Array of shape (n, n) with heuristic values
        """

        # Avoid division by zero
        mask = (distances == 0)
        distances[mask] = 1e-10  

        # Incorporating dynamic socio-economic and environmental factors using random perturbations
        dynamic_socio_economic_factor = np.abs(np.sin(History.iteration / 10) + 1) * 0.5 + 0.5  # Non-linear scaling [0.5, 1.5]
        dynamic_environmental_impact = np.random.rand(*distances.shape) * 0.25  # Scale [0, 0.25]

        # Enhanced heuristic calculation considering non-linear transformation and adaptive response
        attractiveness = np.power((1.0 / distances) * dynamic_socio_economic_factor, 2) * (1 - dynamic_environmental_impact)

        # Normalize the values for better comparability
        attractiveness /= np.sum(attractiveness, axis=1, keepdims=True)

        # Update history with new heuristic values
        History.heuristic.append(attractiveness)

        return attractiveness