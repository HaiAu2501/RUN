import numpy as np

def compute_probabilities(
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    iteration: int,
    n_iterations: int
) -> np.ndarray:
    """Generate unnormalized transition weights using an adaptive approach that balances exploration and exploitation strategies."""
    # Hyperparameters for pheromone and heuristic influences
    alpha = 1.0  # Initial pheromone influence
    beta = 2.0   # Increased emphasis on heuristic influence
    decay_rate = 0.95  # Rate at which pheromone loses its strength
    exploration_factor = 1.5  # Augmenting exploration through enhanced randomness
    noise_factor = 0.1  # Amount of noise to introduce for exploration

    # Adjust pheromone levels dynamically
    pheromone *= decay_rate ** (iteration // (n_iterations // 5))  # Gradual decay over iterations

    # Adjust heuristic weighting based on optimization progress
    adaptive_weight = (1 - (iteration / n_iterations)) * exploration_factor + 1  # Amplifying exploration as iterations progress

    # Compute weights incorporating pheromone and heuristic influences
    weights = np.power(pheromone, alpha) * np.power(heuristic, beta * adaptive_weight)

    # Introduce multiplicative noise to weights for enhanced exploration
    noise = np.random.normal(1, noise_factor, size=weights.shape)  # Noise centered around 1 with a standard deviation defined by noise_factor
    weights *= noise  # Apply randomness to weights for improved exploration

    # Normalize weights to create a valid probability distribution
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    normalized_weights = weights / weights_sum

    return normalized_weights