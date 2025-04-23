import importlib 
import numpy as np
import torch
import tsplib95
import os
import sys
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler

def get_data(name):
	problem = tsplib95.load(f'benchmark/{name}.tsp')

	nodes = list(problem.get_nodes())
	
	coords = np.array([problem.node_coords[node] for node in nodes])
	scaler = MinMaxScaler()
	coords = scaler.fit_transform(coords)

	n = len(nodes)
	distances = np.zeros((n, n))

	for i in range(n):
		for j in range(i + 1, n):
			distance = np.linalg.norm(coords[i] - coords[j])
			distances[i][j] = distance
			distances[j][i] = distance

	optimal = None
	with open('solutions', 'r') as f:
		for line in f:
			line = line.strip()
			if not line or ':' not in line:
				continue
			key, val = line.split(':', 1)
			if key.strip() == name:
				optimal = int(val.strip())
	return distances, optimal, scaler

module = importlib.import_module(sys.argv[1])
name = sys.argv[2]

def run_aco(n_ants=50, n_iterations=200):
    distances, optimal = get_data(name)
    for seed in range(1, 6):
        avg_costs = 0
        aco = module.AntColonyOptimization(
            distances=distances,
            n_ants=n_ants,
            n_iterations=n_iterations,
            seed=seed,
            pheromone_strategy=module.PheromoneImpl(),
            heuristic_strategy=module.HeuristicImpl(),
            probability_strategy=module.ProbabilityImpl()
        )
        cost = aco.run()
        avg_costs += cost
    avg_costs /= 5
    print(f"Average cost for {name}: {avg_costs}")
    print(f"Optimal gap for {name}: {abs(avg_costs - optimal) / optimal * 100:.2f}%")

if __name__ == "__main__":
    run_aco(n_ants=50, n_iterations=200)