import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        # Initial decay and pheromone scale settings
        decay = 0.9
        pheromone_scale = 0.1

        # Adaptive decay adjustment using tanh based on cost trend analysis
        if len(History.costs) > 1:
            prev_cost = History.costs[-2].mean()
            curr_cost = History.costs[-1].mean()
            cost_trend = (curr_cost - prev_cost) / (prev_cost + 1e-5)  # Calculate cost percent change
            decay_factor = torch.tanh(-cost_trend) * 0.1 + 0.85
            decay = max(0.85, min(0.95, decay * decay_factor))

        # Evaporation with adaptive decay
        pheromone *= decay

        # Compute pheromone deposition based on path costs and update routes
        for ant in range(paths.size(1)):
            path = paths[:, ant]  # Path of a given ant
            cost = costs[ant]  # Cost of the path
            deposit_amount = pheromone_scale / (cost + 1e-5)  # Inverse proportional scaling with stability safeguard

            # Deposit to each edge of the path
            for city_idx in range(len(path) - 1):
                i, j = path[city_idx], path[city_idx + 1]
                pheromone[i, j] += deposit_amount
                pheromone[j, i] += deposit_amount

        # Record updates to history for further analysis and adaptive tuning
        History.decay.append(decay)
        History.pheromone.append(pheromone.clone())

        return pheromone
