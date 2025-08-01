import numpy as np
from sklearn.cluster import KMeans

import numpy as np

import numpy as np

def heuristics(distance_matrix):
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix)
    pheromone_levels = np.ones_like(distance_matrix)
    initial_num_ants = 30
    alpha = 1.0
    beta = 4.0
    evaporation_rate = 0.05
    iterations = 100

    def calculate_cost(tour):
        return sum(distance_matrix[tour[i], tour[(i + 1) % num_nodes]] for i in range(num_nodes))

    best_overall_cost = float('inf')
    
    for iteration in range(iterations):
        num_ants = max(int(initial_num_ants * (1 / (1 + np.exp(0.07 * (best_overall_cost - 300))))), 5)
        all_tours = []
        iteration_best_cost = float('inf')
        iteration_best_tour = None

        for _ in range(num_ants):
            tour = [np.random.choice(num_nodes)]
            visited = set(tour)

            while len(tour) < num_nodes:
                current_node = tour[-1]
                probabilities = np.zeros(num_nodes)

                for next_node in range(num_nodes):
                    if next_node not in visited:
                        pheromone_score = pheromone_levels[current_node, next_node]
                        proximity_score = 1 / (distance_matrix[current_node, next_node] + 1e-10)
                        probabilities[next_node] = (pheromone_score ** alpha) * (proximity_score ** beta)

                probabilities /= np.sum(probabilities)
                next_node = np.random.choice(num_nodes, p=probabilities)
                tour.append(next_node)
                visited.add(next_node)

            all_tours.append(tour)
            cost = calculate_cost(tour)
            if cost < iteration_best_cost:
                iteration_best_cost = cost
                iteration_best_tour = tour

        # Local Search (2-opt) for better solutions
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if j != (i + 1) % num_nodes:
                    new_tour = iteration_best_tour[:i + 1] + iteration_best_tour[i+1:j + 1][::-1] + iteration_best_tour[j + 1:]
                    new_cost = calculate_cost(new_tour)
                    if new_cost < iteration_best_cost:
                        iteration_best_cost = new_cost
                        iteration_best_tour = new_tour

        # Pheromone update
        if iteration_best_tour is not None:
            cost_gradient = 1 / (iteration_best_cost + 1e-10)
            for i in range(num_nodes):
                from_node = iteration_best_tour[i]
                to_node = iteration_best_tour[(i + 1) % num_nodes]
                pheromone_levels[from_node, to_node] += cost_gradient * 0.9
                pheromone_levels[to_node, from_node] += cost_gradient * 0.9 

        # Evaporate pheromones
        pheromone_levels *= (1 - evaporation_rate)

        # Track overall best cost
        if iteration_best_cost < best_overall_cost:
            best_overall_cost = iteration_best_cost

    # Populate heuristics_matrix based on pheromone levels and proximity
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                heuristics_matrix[i, j] = (pheromone_levels[i, j] ** alpha) * (1 / (distance_matrix[i, j] + 1e-10) ** beta)

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