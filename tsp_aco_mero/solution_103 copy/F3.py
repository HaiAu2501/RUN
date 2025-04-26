import numpy as np
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    """
    Implementation of pheromone update strategy for ACO TSP.
    Introduces an ecosystem-based pheromone update mechanism that encourages not just individual successes but also inter-path competition and synergy.
    """
    def update(self, pheromone: np.ndarray, paths: np.ndarray, 
              costs: np.ndarray) -> np.ndarray:
        """
        Update pheromone levels based on ant paths and solution costs, incorporating competition and dynamic decay adjustments.
        """
        # Hyperparameters
        decay = 0.8 + (0.1 / (1 + History.iteration))  # Adapt decay rate over time
        competition_weight = 1.0  # Explore stronger competition influence

        # Evaporate existing pheromone
        current_pheromone = pheromone * decay

        # Prepare updates array
        n_ants = paths.shape[1]
        pheromone_updates = np.zeros_like(current_pheromone)

        # Calculate the performance ranking of ants based on costs
        ranked_indices = np.argsort(costs)  # Get indices sorted by cost
        ranked_costs = costs[ranked_indices]  

        # Update pheromones based on ranked contributions
        for rank, i in enumerate(ranked_indices):
            path = paths[:, i]
            cost = ranked_costs[rank]

            # Competitive pheromone contribution inversely proportional to rank
            contribution = competition_weight * (1.0 / cost) * (1.0 / (rank + 1))

            # Pheromone deposit for current ant's path
            next_cities = np.roll(path, shift=-1)
            for j in range(len(path)):
                current_city = path[j]
                next_city = next_cities[j]
                pheromone_updates[current_city, next_city] += contribution
                pheromone_updates[next_city, current_city] += contribution  # Symmetric update

        # Integrate contributions into pheromones
        updated_pheromone = current_pheromone + pheromone_updates

        # Store current pheromone and decay rate into history
        History.pheromone.append(updated_pheromone)
        History.decay.append(decay)

        return updated_pheromone
