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
		for _ in range(n_iterations):
			paths = self.gen_path(require_prob=False)
			costs = self.gen_path_costs(paths)
			
			best_cost, best_idx = costs.min(dim=0)
			if best_cost < self.lowest_cost:
				self.shortest_path = paths[:, best_idx]
				self.lowest_cost = best_cost
			
			self.update_pheronome(paths, costs)

		return self.lowest_cost
	
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

def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    # Basic validation
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    n = distance_matrix.shape[0]
    
    # Normalize the distance matrix to have values between 0 and 1
    max_distance = np.max(distance_matrix) + 1e-10  # Prevent division by zero
    normalized_distance = distance_matrix / max_distance
    
    # Create an indicator matrix for edge inclusion
    edge_inclusion = np.zeros_like(normalized_distance)

    # Create a graph from the distance matrix for centrality measures
    G = nx.from_numpy_array(distance_matrix)

    # Calculate degree centrality
    degree_centrality = np.array(list(nx.degree_centrality(G).values()))

    # Calculate clustering coefficients
    clustering_coeffs = np.array(list(nx.clustering(G).values()))

    # Historical performance data: Placeholder (could be a refined model in practice)
    historical_performance = np.ones_like(normalized_distance)

    # Compute heuristics for edge selection combining multiple factors
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip self-loops
                local_sensitivity = np.exp(-normalized_distance[i][j] * n)
                degree_factor = degree_centrality[i] * degree_centrality[j]
                clustering_factor = clustering_coeffs[i] * clustering_coeffs[j]
                promising_factor = historical_performance[i][j]
                
                # Compute edge importance
                edge_importance = local_sensitivity * degree_factor * clustering_factor * promising_factor
                
                edge_inclusion[i][j] = edge_importance

    # Dynamic adaptive thresholds for edge selection based on rolling window statistics
    threshold = np.percentile(edge_inclusion[edge_inclusion > 0], 60)  # Adjusted percentile for thresholds
    edge_inclusion[edge_inclusion < threshold] = 0

    return edge_inclusion

def heuristics(edge_attr: np.ndarray) -> np.ndarray:
    num_edges = edge_attr.shape[0]
    num_attributes = edge_attr.shape[1]

    heuristic_values = np.zeros_like(edge_attr)

    # Apply feature engineering on edge attributes
    transformed_attr = np.log1p(np.abs(edge_attr))  # Taking logarithm of absolute value of attributes

    # Normalize edge attributes
    scaler = StandardScaler()
    edge_attr_norm = scaler.fit_transform(transformed_attr)

    # Calculate correlation coefficients
    correlation_matrix = np.corrcoef(edge_attr_norm.T)

    # Calculate heuristic value for each edge attribute
    for i in range(num_edges):
        for j in range(num_attributes):
            if edge_attr_norm[i][j] != 0:
                heuristic_values[i][j] = np.exp(-8 * edge_attr_norm[i][j] * correlation_matrix[j][j])

    return heuristic_values

def solve_reevo(dist_mat, n_ants=30, n_iterations=100, seed=0):
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    heu = heuristics(dist_mat.copy()) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, heu, n_ants=n_ants, seed=seed)
    obj = aco.run(n_iterations)
    return obj

size = sys.argv[1]
n_ants = 50
n_iterations = 200

def run_reevo(size):
	# avg_costs = 0
    # # Lấy tất cả các file trong thư mục benchmark
	# path = f"tsp_aco_mero/ls_tsp/TSP{size}.npy"
	# prob_batch = np.load(path)
	# from scipy.spatial import distance_matrix
	# for i, prob in enumerate(prob_batch):
	# 	print(f"Processing TSP{size} {i}")
	# 	distances = distance_matrix(prob, prob)
	# 	obj = solve_reevo(distances, n_ants=n_ants, n_iterations=n_iterations, seed=0)
	# 	print(f"Cost for TSP{size} {i}: {obj}")
	# 	avg_costs += obj

	# avg_costs /= len(prob_batch)
	# print(f"Average cost for TSP{size}: {avg_costs}")

	avg_costs = 0
	for i in range(1, 65):
		path = f"tsp_aco_mero/test/TSP{size}_{i:02}.npy"
		distances = np.load(path)
		obj = solve_reevo(distances, n_ants=n_ants, n_iterations=n_iterations, seed=0)
		avg_costs += obj
	print(f"Average cost for TSP{size}: {avg_costs / 64}")

if __name__ == "__main__":
    print(f"Running ACO for TSP{size} with n_ants={n_ants} and n_iterations={n_iterations}")
    run_reevo(size)
