import torch
from torch.distributions import Categorical
import numpy as np
from sklearn.preprocessing import StandardScaler
import tsplib95
import os
import sys
from tqdm import tqdm

class ACO():
	def __init__(self, 
				distances,
				heuristic,
				n_ants=30, 
				decay=0.9,
				alpha=1,
				beta=1,
				device='cpu',
				seed=0
				):
		
		self.problem_size = len(distances)
		self.distances  = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
		self.n_ants = n_ants
		self.decay = decay
		self.alpha = alpha
		self.beta = beta
		
		self.pheromone = torch.ones_like(self.distances)
		self.heuristic = torch.tensor(heuristic, device=device) if not isinstance(heuristic, torch.Tensor) else heuristic

		self.shortest_path = None
		self.lowest_cost = float('inf')

		self.device = device
		self.seed = seed

	@torch.no_grad()
	def run(self, n_iterations):
		torch.manual_seed(seed=self.seed)
		list_cost = []
		for it in range(n_iterations + 1):
			paths = self.gen_path(require_prob=False)
			costs = self.gen_path_costs(paths)
			
			best_cost, best_idx = costs.min(dim=0)
			if best_cost < self.lowest_cost:
				self.shortest_path = paths[:, best_idx]
				self.lowest_cost = best_cost
			
			self.update_pheronome(paths, costs)
			if it % 20 == 0:
				list_cost.append(self.lowest_cost)

		return self.lowest_cost, list_cost
	
	@torch.no_grad()
	def update_pheronome(self, paths, costs):
		'''
		Args:
			paths: torch tensor with shape (problem_size, n_ants)
			costs: torch tensor with shape (n_ants,)
		'''
		self.pheromone = self.pheromone * self.decay 
		for i in range(self.n_ants):
			path = paths[:, i]
			cost = costs[i]
			self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
			self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost

	@torch.no_grad()
	def gen_path_costs(self, paths):
		'''
		Args:
			paths: torch tensor with shape (problem_size, n_ants)
		Returns:
				Lengths of paths: torch tensor with shape (n_ants,)
		'''
		assert paths.shape == (self.problem_size, self.n_ants)
		u = paths.T # shape: (n_ants, problem_size)
		v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
		assert (self.distances[u, v] > 0).all()
		return torch.sum(self.distances[u, v], dim=1)

	def gen_path(self, require_prob=False):
		'''
		Tour contruction for all ants
		Returns:
			paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
			log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
		'''
		start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
		mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
		mask[torch.arange(self.n_ants, device=self.device), start] = 0
		
		paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
		paths_list.append(start)
		
		log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
		
		prev = start
		for _ in range(self.problem_size-1):
			actions, log_probs = self.pick_move(prev, mask, require_prob)
			paths_list.append(actions)
			if require_prob:
				log_probs_list.append(log_probs)
				mask = mask.clone()
			prev = actions
			mask[torch.arange(self.n_ants, device=self.device), actions] = 0
		
		if require_prob:
			return torch.stack(paths_list), torch.stack(log_probs_list)
		else:
			return torch.stack(paths_list)
		
	def pick_move(self, prev, mask, require_prob):
		'''
		Args:
			prev: tensor with shape (n_ants,), previous nodes for all ants
			mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
		'''
		pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
		heuristic = self.heuristic[prev] # shape: (n_ants, p_size)
		dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask) # shape: (n_ants, p_size)
		dist = Categorical(dist)
		actions = dist.sample() # shape: (n_ants,)
		log_probs = dist.log_prob(actions) if require_prob else None # shape: (n_ants,)
		return actions, log_probs


import numpy as np
import networkx as nx

def heuristics(distance_matrix):
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and distance_matrix[i, j] > 0:
                # Heuristic: inverse distance cubed, with a penalty for longer distances
                heuristics_matrix[i, j] = 1 / (distance_matrix[i, j] ** 3 + 1e-10) * (1 / (np.mean(distance_matrix[i]) + 1e-10))

    return heuristics_matrix

def solve_reevo(dist_mat, n_ants=30, n_iterations=100, seed=1234):
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    heu = heuristics(dist_mat.copy()) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, heu, n_ants=n_ants, seed=seed)
    obj, cost = aco.run(n_iterations)
    return obj, cost

from scipy.spatial import distance_matrix

size = sys.argv[1]

if size in ["20", "50", "100"]:
	N_ANTS = 50
	N_ITERATIONS = 200
else:
	N_ANTS = 100
	N_ITERATIONS = 200

def run_reevo(seed=0):
	costs = []
	path = f"tsp_aco/datasets/test_TSP{size}.npy"
	batch = np.load(path)
	for i, prob in enumerate(batch):
		distances = distance_matrix(prob, prob)
		obj, cost = solve_reevo(distances, n_ants=N_ANTS, n_iterations=N_ITERATIONS, seed=seed)
		costs.append(cost)
	mean_costs = np.mean(np.array(costs), axis=0)
	print(mean_costs.tolist())

if __name__ == "__main__":
    print(f"Running ACO for TSP{size} with n_ants={N_ANTS} and n_iterations={N_ITERATIONS}")
    run_reevo(seed=0)
