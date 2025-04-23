import torch
from .mero import *

class PheromoneImpl(PheromoneStrategy):
    def update(self, pheromone: torch.Tensor, paths: torch.Tensor, 
              costs: torch.Tensor) -> torch.Tensor:
        if len(History.costs) > 1:
            last_cost = History.costs[-1].mean()
            second_last_cost = History.costs[-2].mean()
            improvement = (second_last_cost - last_cost) / second_last_cost
            adaptive_decay = max(0.7, 0.9 - 0.3 * torch.clamp(improvement, 0.0, 0.2))
        else:
            adaptive_decay = 0.9

        # Track adaptive_decay
        History.decay.append(adaptive_decay)

        # Ensure minimum pheromone level for numerical stability
        epsilon = 1e-10
        pheromone.fill_(pheromone.max() * epsilon)

        # Evaporate current pheromone
        pheromone *= adaptive_decay

        # Deposit Pheromone Based on Paths and Costs
        for path, cost in zip(paths.T, costs):
            reward = max(0, (1.0 / cost) - (1.0 / costs.mean()))
            for u, v in zip(path, torch.roll(path, shifts=-1)):
                pheromone[u, v] += reward
                pheromone[v, u] += reward

        # Combine with Historical Pheromone Levels
        if len(History.pheromone) > 0:
            prev_pheromone = History.pheromone[-1]
            pheromone = 0.5 * pheromone + 0.5 * prev_pheromone

        # Update History with New Pheromone Levels
        History.pheromone.append(pheromone.clone())

        return pheromone
