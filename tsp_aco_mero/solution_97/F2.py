import torch
from .mero import *

class ProbabilityImpl(ProbabilityStrategy):
    def compute(self, pheromone: np.ndarray, heuristic: np.ndarray) -> np.ndarray:
        # Dynamic temperature based on iteration number to adjust influence of historical vs. heuristic data
        max_iterations = 100  # Define maximum iterations for dynamic temperature adjustment
        T_t = max(1e-1, 1.0 - (History.iteration / max_iterations))

        # Prevent log(0) error by adding a small constant
        pheromone_stable = np.maximum(pheromone, 1e-10)
        heuristic_stable = np.maximum(heuristic, 1e-10)

        # Combined influence using logarithmic scaling (for numerical stability)
        combined_influence = np.log(pheromone_stable) / T_t + np.log(heuristic_stable) / T_t

        # Calculate softmax probabilities on the combined influence
        exp_influence = np.exp(combined_influence - np.max(combined_influence))  # Stability improvement
        probabilities = exp_influence / np.sum(exp_influence)

        # Consistently track history for verification and debugging
        History.alpha.append(np.mean(History.alpha) if History.alpha else 1.0)
        History.beta.append(np.mean(History.beta) if History.beta else 1.0)  

        return probabilities
