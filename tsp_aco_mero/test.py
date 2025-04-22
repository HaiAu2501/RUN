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

def run_aco(size):
    avg_costs = 0
    if size == "200":
        n_ants = 100
        n_iterations = 200
    elif size == "500":
        n_ants = 100
        n_iterations = 500
    elif size == "1000":
        n_ants = 100
        n_iterations = 1000
    path = f"tsp_aco_mero/ls_tsp/TSP{size}.npy"
    prob_batch = np.load(path)
    for i, prob in tqdm(enumerate(prob_batch), desc=f"Processing TSP{size}", start=1):
        distances = prob
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
        avg_costs += cost
    avg_costs /= len(prob_batch)
    print(f"Average cost for TSP{size}: {avg_costs}")

if __name__ == "__main__":
    run_aco(size)