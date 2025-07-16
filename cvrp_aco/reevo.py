import torch
from torch.distributions import Categorical
import random
import itertools
import numpy as np



class ACO():

    def __init__(self,  # 0: depot
                 distances, # (n, n)
                 demand,   # (n, )
                 heuristic, # (n, n)
                 capacity,
                 n_ants=30, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu',
                 seed=0,
                 ):
        
        self.problem_size = len(distances)
        self.distances = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
        self.demand = torch.tensor(demand, device=device) if not isinstance(demand, torch.Tensor) else demand
        self.capacity = capacity
                
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
        torch.manual_seed(self.seed)
        list_costs = []
        for it in range(n_iterations + 1):
            paths = self.gen_path()
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
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
        self.pheromone[self.pheromone < 1e-10] = 1e-10
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self):
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)
        
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        
        paths_list = [actions] # paths_list[i] is the ith move (tensor) for all ants
        
        done = self.check_done(visit_mask, actions)
        while not done:
            actions = self.pick_move(actions, visit_mask, capacity_mask)
            paths_list.append(actions)
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            done = self.check_done(visit_mask, actions)
            
        return torch.stack(paths_list)
        
    def pick_move(self, prev, visit_mask, capacity_mask):
        pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev] # shape: (n_ants, p_size)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask * capacity_mask) # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample() # shape: (n_ants,)
        return actions
    
    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask
    
    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size) # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants, 1) # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        
        return used_capacity, capacity_mask
    
    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()    

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    
    n = distance_matrix.shape[0]
    promising_matrix = np.zeros_like(distance_matrix)

    # Clustering based on coordinates for adaptive routing
    clustering = DBSCAN(eps=1.0, min_samples=1).fit(coordinates)
    labels = clustering.labels_

    # Calculate total demand per cluster for demand density
    cluster_demand = np.zeros(len(set(labels)))
    for i in range(1, n):  # Skip depot
        cluster_demand[labels[i]] += demands[i]

    # Calculate average distances within and between clusters
    cluster_centers = np.array([coordinates[labels == label].mean(axis=0) for label in set(labels)])
    cluster_distances = cdist(cluster_centers, cluster_centers)

    # Compute promising scores based on multiple objectives
    for i in range(n):
        for j in range(n):
            if i != j and distance_matrix[i, j] > 0:
                if demands[j] <= capacity:
                    # Demand to distance ratio
                    demand_distance_ratio = demands[j] / distance_matrix[i, j]

                    # Demand density influence
                    cluster_load_ratio = cluster_demand[labels[j]] / capacity

                    # Efficiency based on intra-cluster distances
                    efficiency_penalty = 1 - (cluster_distances[labels[i]][labels[j]] / (np.max(cluster_distances) + 1e-10))

                    # Load balance consideration
                    remaining_capacity = capacity - demands[j]
                    load_balance_factor = remaining_capacity / capacity if remaining_capacity > 0 else 0

                    # Historical performance adjustments
                    historical_penalty_factor = 1 / (1 + np.square(distance_matrix[i, j]))

                    # Calculate combined promising score
                    promising_score = (demand_distance_ratio * 
                                       cluster_load_ratio * 
                                       efficiency_penalty * 
                                       load_balance_factor * 
                                       historical_penalty_factor)

                    # Set the promising score in the matrix
                    promising_matrix[i, j] = promising_score

    # Sparsify the score matrix by zeroing out low-promising scores
    promising_matrix[promising_matrix < 0.01] = 0  # Threshold for promising edges

    # Normalize promising scores for comparability
    max_score = np.max(promising_matrix)
    if max_score > 0:
        promising_matrix = promising_matrix / max_score  # Normalize to [0, 1]
    
    return promising_matrix


from scipy.spatial import distance_matrix
import inspect

N_ITERATIONS = 100
N_ANTS = 30
CAPACITY = 50

def solve(node_pos, demand):
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    if len(inspect.getfullargspec(heuristics).args) == 4:
        heu = heuristics(dist_mat.copy(), node_pos.copy(), demand.copy(), CAPACITY) + 1e-9
    elif len(inspect.getfullargspec(heuristics).args) == 2:
        heu = heuristics(dist_mat.copy(), demand / CAPACITY) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, demand, heu, CAPACITY, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return obj

import os, sys

def run(size):
    print(f"Running CVRP-ACO [EoH] for CVRP{size}...")
    path = os.path.join(os.path.dirname(__file__), 'datasets', f'test_CVRP{size}.npy')
    dataset = np.load(path)
    demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
    n_instances = node_positions.shape[0]
    LIST = []
    for i, (node_pos, demand) in enumerate(zip(node_positions, demands)):
        obj = solve(node_pos, demand)
        LIST.append(obj)

    LIST = np.array(LIST)
    print(LIST.mean(axis=0).tolist())

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(size)