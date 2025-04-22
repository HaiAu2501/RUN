import importlib 
import numpy as np
import torch
import tsplib95
import os
import sys
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod


module = importlib.import_module(sys.argv[1])

def run_aco(n_ants=30, n_iterations=100):
    # Lấy tất cả các file trong thư mục benchmark
    for size in [20, 50, 100]:
        avg_costs = 0
        for i in range(1, 65):
            path = f"tsp_aco_mero/test/TSP{size}_{i:02}.npy"
            distances = np.load(path)
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
        avg_costs /= 64
        print(f"Average cost for TSP{size}: {avg_costs}")

if __name__ == "__main__":
    run_aco(n_ants=50, n_iterations=200)