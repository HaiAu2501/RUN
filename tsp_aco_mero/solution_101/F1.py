import numpy as np
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    """
    Enhanced implementation of heuristic strategy for ACO TSP.
    Utilizes dynamic scoring systems which integrate multi-layered contextual factors to transform distances into heuristic attractiveness values.
    """

    def compute(self, distances: np.ndarray, 
                attractiveness_scores: np.ndarray = None, 
                socio_economic_data: np.ndarray = None) -> np.ndarray:
        """
        Convert distances to heuristic attractiveness values by enhancing traditional metrics with diverse contextual information.

        Args:
            distances: Array of shape (n, n) with distances between cities.
            attractiveness_scores: Optional array of shape (n, n) with scores based on qualitative factors.
            socio_economic_data: Optional array of shape (n, n) representing socio-economic indicators for attractiveness enhancements.

        Returns:
            Array of shape (n, n) with transformed heuristic values.
        """
        # Avoid division by zero
        mask = (distances == 0)
        distances[mask] = 1e-10  

        # Step 1: Compute nonlinear inverse distance transformation (quadratic scaling to emphasize larger distances)
        nonlin_inverse_distances = 1.0 / (distances ** 2)

        # Step 2: Initialize adjusted scores
        adjusted_scores = np.zeros_like(nonlin_inverse_distances)
        
        # Step 3: Normalize attractiveness scores if provided
        if attractiveness_scores is not None:
            attractiveness_scores = np.clip(attractiveness_scores, 0, None)
            normalized_attractiveness = attractiveness_scores / np.sum(attractiveness_scores, axis=1, keepdims=True)
            adjusted_scores += normalized_attractiveness  

        # Step 4: Enhance with socio-economic data (if provided)
        if socio_economic_data is not None:
            socio_economic_data = np.clip(socio_economic_data, 0, None)
            normalized_socio_economic = socio_economic_data / np.sum(socio_economic_data, axis=1, keepdims=True)
            adjusted_scores += normalized_socio_economic

        # Step 5: Combine adjusted scores with non-linear inverse distances, applying adaptive weights
        weights = np.array([0.6, 0.4])  # Allows more emphasis on distance
        transformed_values = (weights[0] * nonlin_inverse_distances) + (weights[1] * adjusted_scores)

        # Step 6: Track heuristic values in history
        History.heuristic.append(transformed_values)

        return transformed_values