import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    def compute(self, distances: np.ndarray) -> np.ndarray:
        # Avoid division by zero
        mask = (distances == 0)
        distances_safe = np.where(mask, 1e-10, distances)

        # Define an adaptable temperature parameter T
        T = 1.0 / (History.iteration + 1)  # Decreases with iterations
        
        # Calculate the attractiveness values using the proposed formula
        attractiveness = (1.0 / distances_safe) * np.exp(-distances_safe / T)

        # Normalize attractiveness values
        attractiveness_max = np.max(attractiveness)  # For normalization
        attractiveness = np.clip(attractiveness / attractiveness_max, 0, 1)  # Scale to [0, 1]

        # Record current heuristic for historical tracking
        History.heuristic.append(attractiveness)

        return attractiveness

