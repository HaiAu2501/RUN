import importlib 
import numpy as np
import torch
import tsplib95
import os
import sys
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod
from tqdm import tqdm


module = importlib.import_module(sys.argv[1])
size = sys.argv[2]
n_ants = 100
n_iterations = 200

def run_aco(size):
    avg_costs = 0
    path = f"tsp_aco_mero/ls_tsp/TSP{size}.npy"
    prob_batch = np.load(path)
    from scipy.spatial import distance_matrix
    # Calculate the distance matrix
    for i, prob in enumerate(prob_batch):
        print(f"Processing TSP{size} {i}")
        distances = distance_matrix(prob, prob)
        aco = module.AntColonyOptimization(
            distances=distances,
            n_ants=n_ants,
            n_iterations=n_iterations,
            seed=0,
            pheromone_strategy=module.PheromoneImpl(),
            heuristic_strategy=module.HeuristicImpl(),
            probability_strategy=module.ProbabilityImpl()
        )
        cost = aco.run()
        print(f"Cost for TSP{size} {i}: {cost}")
        avg_costs += cost
    avg_costs /= len(prob_batch)
    print(f"Average cost for TSP{size}: {avg_costs}")

if __name__ == "__main__":
    print(f"Running ACO for TSP{size} with n_ants={n_ants} and n_iterations={n_iterations}")
    run_aco(size)