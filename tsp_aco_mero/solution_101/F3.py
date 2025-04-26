import numpy as np
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Emphasizes collective adaptive behavior, dynamic pheromone layers, and
    adaptive decay rates to enhance exploration and solution discovery.
    """
    def update(self, pheromone: np.ndarray, paths: np.ndarray, 
              costs: np.ndarray) -> np.ndarray:
        """
        Update pheromone levels based on ant paths and solution costs.
        
        Args:
            pheromone: Array of shape (n, n) with current pheromone levels
            paths: Array of shape (n_cities, n_ants) with paths taken by ants
            costs: Array of shape (n_ants,) with path costs
            
        Returns:
            Updated pheromone array
        """
        # Hyperparameters
        decay = np.mean(History.decay) if History.decay else 0.85  # Adjusted dynamic decay based on history
        exploration_rate = 0.2  # Encourages random exploration
        pheromone_strength = 100.0 / (np.mean(costs) + 1e-6)  # Normalize pheromone contributions

        # Dynamic pheromone decay based on collective learning 
        recent_costs = costs / np.mean(costs)  # Relative performance metrics
        collective_performance = np.mean(recent_costs)  # Assess group dynamics
        # Modify decay based on collective performance
        if collective_performance < 1:
            decay *= (1.0 + (1 - collective_performance))  # Increase decay rate if collective performance worsens
        else:
            decay *= (1.0 - (collective_performance - 1))  # Decrease decay rate if collective performance improves

        # Apply evaporation with historical decay rate
        pheromone *= decay

        # Normalize costs for better pheromone deposit weight
        norm_costs = costs - np.min(costs) + 1e-6  # Avoid division by zero
        weights = (pheromone_strength / norm_costs)
        weights /= np.sum(weights)  # Normalize weights to sum to 1

        # Pheromone deposit based on paths and collective performance
        n_ants = paths.shape[1]
        for i in range(n_ants):
            path = paths[:, i]
            weight = weights[i]

            # Add pheromone to edges in the path
            next_cities = np.roll(path, shift=-1)
            for j in range(len(path)):
                current_city = path[j]
                next_city = next_cities[j]
                pheromone[current_city, next_city] += weight
                pheromone[next_city, current_city] += weight  # For symmetric TSP

        # Probabilistic boost for a fraction of the paths based on performance
        for idx in np.random.choice(range(n_ants), size=int(n_ants * exploration_rate), replace=False):
            path = paths[:, idx]
            for city in path:
                pheromone[city] += 0.1 / len(path)  # Boosting selected paths

        # Track updated hyperparameters in history
        History.decay.append(decay)

        return pheromone
