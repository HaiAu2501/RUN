# Adapted from `mkp_transformer` in [*DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization*](https://github.com/henry-yeh/DeepACO)

import torch
from torch.distributions import Categorical
import numpy as np

class ACO():

    def __init__(self,  # constraints are set to 1 after normalize weight 
                 prize,  # shape [n,]
                 weight, # shape [m, n]
                 heuristic,
                 n_ants=30, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu',
                 seed=0,
                 ):
        self.n, self.m = weight.shape
        
        self.prize = prize
        self.weight = weight
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        self.pheromone = torch.ones(size=(self.n+1,), device=device)

        # Fidanova S. Hybrid ant colony optimization algorithm for multiple knapsack problem
        # self.heuristic = prize / self.weight.sum(dim=1) if heuristic is None else heuristic
        self.heuristic = heuristic
        # Leguizamon G, Michalewicz Z. A New Version of Ant System for Subset Problems
        self.Q = 1 / self.prize.sum()

        self.alltime_best_sol = None
        self.alltime_best_obj = 0
        self.device = device
        self.add_dummy_node()
        self.seed = seed
        
    def add_dummy_node(self):
        self.prize = torch.cat((self.prize, torch.tensor([0.], device=self.device))) # (n+1,)
        self.weight = torch.cat((self.weight, torch.zeros((1, self.m), device=self.device)), dim=0) # (n+1, m)
        self.heuristic = torch.cat((self.heuristic, torch.tensor([1e-8], device=self.device))) # (n+1)

    @torch.no_grad()
    def run(self, n_iterations):
        torch.manual_seed(self.seed)
        list_objs = []
        for it in range(n_iterations + 1):
            sols = self.gen_sol() # (n_ants, max_horizon)
            objs = self.gen_sol_obj(sols)             # (n_ants,)
            sols = sols.T
            best_obj, best_idx = objs.max(dim=0)
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx]
            self.update_pheronome(sols, objs, best_obj.item(), best_idx.item())
            if it % 5 == 0:
                list_objs.append(self.alltime_best_obj.item())
        return list_objs

    @torch.no_grad()
    def update_pheronome(self, sols, objs, best_obj, best_idx):
        self.pheromone = self.pheromone * self.decay 
        for i in range(self.n_ants):
            sol = sols[i]
            obj = objs[i]
            self.pheromone[sol] += self.Q * obj

    @torch.no_grad()
    def gen_sol_obj(self, solutions):
        '''
        Args:
            solutions: (n_ants, max_horizon)
        Return:
            obj: (n_ants,)
        '''
        return self.prize[solutions.T].sum(dim=1) # (n_ants,)

    def gen_sol(self):
        '''
        Solution contruction for all ants
        '''
        solutions = [] # solutions[i] is the i-th picked item for all ants
        knapsack = torch.zeros(size=(self.n_ants, self.m), device=self.device)  # used capacity
        mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)
        dummy_mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)
        dummy_mask[:, -1] = 0
        
        mask, knapsack = self.update_knapsack(mask, knapsack, new_item=None)
        dummy_mask = self.update_dummy_state(mask, dummy_mask)
        done = self.check_done(mask)
        while not done:
            items = self.pick_item(mask, dummy_mask)
            solutions.append(items)
            mask, knapsack = self.update_knapsack(mask, knapsack, items)
            dummy_mask = self.update_dummy_state(mask, dummy_mask)
            done = self.check_done(mask)
        return torch.stack(solutions)
    
    def pick_item(self, mask, dummy_mask):
        phe = self.pheromone.unsqueeze(0).repeat(self.n_ants, 1)
        heu = self.heuristic.unsqueeze(0).repeat(self.n_ants, 1)
        dist = ((phe ** self.alpha) * (heu ** self.beta) * mask * dummy_mask) # (n_ants, n+1)
        dist = Categorical(dist)
        item = dist.sample()
        return item # (n_ants,)
    
    def check_done(self, mask):
        # is mask all zero except for the dummy node?
        return (mask[:, :-1] == 0).all()
    
    def update_dummy_state(self, mask, dummy_mask):
        finished = (mask[: ,:-1] == 0).all(dim=1)
        dummy_mask[finished] = 1
        return dummy_mask
    
    def update_knapsack(self, mask, knapsack, new_item):
        '''
        Args:
            mask: (n_ants, n+1)
            knapsack: (n_ants, m)
            new_item: (n_ants)
        '''
        if new_item is not None:
            mask[torch.arange(self.n_ants), new_item] = 0
            knapsack += self.weight[new_item] # (n_ants, m)
        for ant_idx in range(self.n_ants):
            candidates = torch.nonzero(mask[ant_idx]) # (x, 1)
            if len(candidates) > 1:
                candidates.squeeze_()
                test_knapsack = knapsack[ant_idx].unsqueeze(0).repeat(len(candidates), 1) # (x, m)
                new_knapsack = test_knapsack + self.weight[candidates] # (x, m)
                infeasible_idx = candidates[(new_knapsack > 1).any(dim=1)]
                mask[ant_idx, infeasible_idx] = 0
        mask[:, -1] = 1
        return mask, knapsack
    
########################

import numpy as np

def heuristics(prize, weight):
    n = prize.shape[0]
    m = weight.shape[1]
    heuristics_matrix = np.zeros(n)
    
    for i in range(n):
        if np.all(weight[i] <= 1):  # Ensure the weight constraints are met
            heuristics_matrix[i] = prize[i] / np.max(weight[i])  # Normalize by the maximum weight component
            
    return heuristics_matrix

####################################

N_ITERATIONS = 50
N_ANTS = 10

def solve(prize: np.ndarray, weight: np.ndarray):
    n, m = weight.shape
    heu = heuristics(prize.copy(), weight.copy()) + 1e-9
    assert heu.shape == (n,)
    heu[heu < 1e-9] = 1e-9
    aco = ACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(heu), N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return obj

import os, sys

def run(size):
    """
    Run the ACO algorithm for the MKP problem.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", f"test_MKP{size}.npz")
    dataset = np.load(dataset_path)
    prizes, weights = dataset['prizes'], dataset['weights']
    
    objs = []
    for i, (prize, weight) in enumerate(zip(prizes, weights)):
        obj = solve(prize, weight)
        objs.append(obj)

    objs = np.array(objs)
    print(objs.mean(axis=0).tolist())

if __name__ == "__main__":
    # Get size from command line argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(size)