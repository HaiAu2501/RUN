import numpy as np
from sklearn.cluster import KMeans

def heuristics(distance_matrix):
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.zeros((num_nodes, num_nodes))

    # Step 1: Clustering nodes into groups
    num_clusters = max(2, num_nodes // 2)  # At least two clusters
    kmeans = KMeans(n_clusters=num_clusters)
    node_indices = np.arange(num_nodes).reshape(-1, 1)
    kmeans.fit(node_indices)
    labels = kmeans.labels_

    # Step 2: Calculate average distances and cubic inverse distances
    for i in range(num_nodes):
        valid_distances = distance_matrix[i][distance_matrix[i] > 0]
        avg_distance = np.mean(valid_distances) if len(valid_distances) > 0 else 0
        inverse_cubic_distance = 1 / (distance_matrix[i] ** 3 + 1e-6)  # Avoid division by zero
        
        for j in range(num_nodes):
            if i != j:
                cluster_weight = (labels[i] == labels[j]) * 0.6 + 0.4  # Weight for intra-cluster connections
                distance_score = inverse_cubic_distance[j] * (avg_distance - distance_matrix[i][j]) / (avg_distance + 1e-6) if avg_distance > 0 else 0
                heuristics_matrix[i][j] = max(distance_score * cluster_weight, 0)

    # Step 3: Adaptive normalization
    row_sum = heuristics_matrix.sum(axis=1, keepdims=True)
    heuristics_matrix = heuristics_matrix / (row_sum + 1e-6)  # Normalize with small constant to avoid zero division

    return heuristics_matrix

import torch
from torch.distributions import Categorical
import numpy as np
from sklearn.preprocessing import StandardScaler
import tsplib95
import os
import sys
from tqdm import tqdm
from scipy.spatial import distance_matrix

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
		list_costs = []
		for it in range(n_iterations + 1):
			paths = self.gen_path(require_prob=False)
			costs = self.gen_path_costs(paths)
			
			best_cost, best_idx = costs.min(dim=0)
			if best_cost < self.lowest_cost:
				self.shortest_path = paths[:, best_idx]
				self.lowest_cost = best_cost
			
			self.update_pheronome(paths, costs)

			if it % 10 == 0:
				list_costs.append(self.lowest_cost.item())

		return list_costs
	
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

def solve(dist_mat, n_ants=30, n_iterations=100, seed=0):
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    heu = heuristics(dist_mat.copy()) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, heu, n_ants=n_ants, seed=seed)
    obj = aco.run(n_iterations)
    return obj

def run(size):
	print(f"Running TSP-ACO-EoH for TSP{size}...")
	path = os.path.join(os.path.dirname(__file__), 'datasets', f'test_TSP{size}.npy')
	data = np.load(path)  # shape (n_instances, size, 2)
	results = []
	n_instances = data.shape[0]
	for i in range(n_instances):
		coords = data[i]
		dist_mat = distance_matrix(coords, coords)
		costs = solve(dist_mat, n_ants=50, n_iterations=100, seed=i)
		results.append(costs)
	return np.array(results)

if __name__ == "__main__":
	size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
	results = run(size)
	mean_costs = np.mean(results, axis=0)  # Average costs over all instances
	print(mean_costs.tolist())