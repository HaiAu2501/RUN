import torch
from .mero import *

class HeuristicImpl(HeuristicStrategy):
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        # Step 1: Handle zero distances
        safe_distances = torch.where(distances == 0, 1e-10, distances)
        
        # Step 2: Compute inverse scaled heuristics
        inverse_scaled = 1.0 / safe_distances
        heuristic_values = inverse_scaled / inverse_scaled.sum(dim=1, keepdim=True)

        # Step 3: Dynamic feedback based scaling
        if History.pheromone:
            pheromone_levels = History.pheromone[-1]
            avg_phero = pheromone_levels.mean()
            max_phero = pheromone_levels.max()

            scaling_factor = avg_phero / (max_phero + 1e-10) * self.dynamic_scaling_threshold()
            heuristic_values *= scaling_factor

        # Clamp heuristic values
        heuristic_values = heuristic_values.clamp(min=1e-10, max=1.0)

        # Track heuristic values
        History.heuristic.append(heuristic_values.clone())

        return heuristic_values

    @staticmethod
    def dynamic_scaling_threshold() -> float:
        # Dummy performance feedback implementation
        # Assuming some performance metric is stored in History.costs for illustration
        if History.costs:
            iteration_costs = History.costs[-1]
            # Hypothetical feedback calculation
            performance_ratio = 1.0 / (torch.mean(iteration_costs) + 1e-10)
            return 0.1 + 0.9 * performance_ratio
        return 1.0