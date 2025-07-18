import torch
from torch.distributions import Categorical

class ACO():

    def __init__(self,
                 prizes,
                 distances,
                 max_len,
                 heuristic,
                 n_ants=20,
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu',
                 seed=0,
                 ):
        
        self.n = len(prizes)
        self.distances = distances
        self.prizes = prizes
        self.max_len = max_len
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone = torch.ones_like(self.distances)
        self.heuristic = heuristic
        
        self.Q = 1 / prizes.sum()
        
        self.alltime_best_sol = None
        self.alltime_best_obj = 0

        self.device = device
        
        self.add_dummy_node()

        self.seed = seed
        
    def add_dummy_node(self):
        '''
        One has to sparsify the graph first before adding dummy node
        distance: 
                [[1e9 , x   , x   , 0  ],
                [x   , 1e9 , x   , 0  ],
                [x   , x   , 1e9 , 0  ],
                [1e10, 1e10, 1e10, 0  ]]
        pheromone: [1]
        heuristic: [>0]
        prizes: [x,x,...,0]
        '''
        self.prizes = torch.cat((self.prizes, torch.tensor([1e-10], device=self.device)))
        distances = torch.cat((self.distances, 1e10 * torch.ones(size=(1, self.n), device=self.device)), dim=0)
        self.distances = torch.cat((distances, 1e-10 + torch.zeros(size=(self.n+1, 1), device=self.device)), dim=1)

        self.heuristic = torch.cat((self.heuristic, torch.zeros(size=(1, self.n), device=self.device)), dim=0) # cannot reach other nodes from dummy node
        self.heuristic = torch.cat((self.heuristic, torch.ones(size=(self.n+1, 1), device=self.device)), dim=1)

        self.pheromone = torch.ones_like(self.distances)
        self.distances[self.distances == 1e-10] = 0
        self.prizes[-1] = 0

    @torch.no_grad()
    def run(self, n_iterations):
        torch.manual_seed(self.seed)
        list_obj = []
        for it in range(n_iterations + 1):
            sols = self.gen_sol()
            objs = self.gen_sol_obj(sols)
            sols = sols.T
            best_obj, best_idx = objs.max(dim=0)
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx]
            self.update_pheronome(sols, objs, best_obj, best_idx)
            if it % 10 == 0:
                list_obj.append(self.alltime_best_obj.item())
        return list_obj
       
    
    @torch.no_grad()
    def update_pheronome(self, sols, objs, best_obj, best_idx):
        self.pheromone = self.pheromone * self.decay
        for i in range(self.n_ants):
            sol = sols[i]
            obj = objs[i]
            self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] += self.Q * obj
                
    
    @torch.no_grad()
    def gen_sol_obj(self, solutions):
        '''
        Args:
            solutions: (max_len, n_ants)
        '''
        objs = self.prizes[solutions.T].sum(dim=1)
        return objs

    def gen_sol(self):
        '''
        Solution contruction for all ants
        '''
        solutions = []

        solutions = [torch.zeros(size=(self.n_ants,), device=self.device, dtype=torch.int64)]
        mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)
        done = torch.zeros(size=(self.n_ants,), device=self.device)
        travel_dis = torch.zeros(size=(self.n_ants,), device=self.device)
        cur_node = torch.zeros(size=(self.n_ants,), dtype=torch.int64, device=self.device)
        
        mask = self.update_mask(travel_dis, cur_node, mask)
        done = self.check_done(mask)
        # construction
        while not done:
            nxt_node = self.pick_node(mask, cur_node) # pick action
            # update solution and log_probs
            solutions.append(nxt_node) 
            # update travel_dis, cur_node and mask
            travel_dis += self.distances[cur_node, nxt_node]
            cur_node = nxt_node
            mask = self.update_mask(travel_dis, cur_node, mask)
            # check done
            done = self.check_done(mask)
        return torch.stack(solutions)
    
    def pick_node(self, mask, cur_node):
        pheromone = self.pheromone[cur_node] # shape: (n_ants, p_size+1)
        heuristic = self.heuristic[cur_node] # shape: (n_ants, p_size+1)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask)
        dist = Categorical(dist)
        item = dist.sample()
        return item  # (n_ants,)
    
    def update_mask(self, travel_dis, cur_node, mask):
        '''
        Args:
            travel_dis: (n_ants,)
            cur_node: (n_ants,)
            mask: (n_ants, n+1)
        '''
        mask[torch.arange(self.n_ants), cur_node] = 0

        for ant_id in range(self.n_ants):
            if cur_node[ant_id] != self.n: # if not at dummy node
                _mask = mask[ant_id]
                candidates = torch.nonzero(_mask).squeeze()
                # after going to candidate node from cur_node, can it return to depot?
                trails = travel_dis[ant_id] + self.distances[cur_node[ant_id], candidates] + self.distances[candidates, 0]
                fail_idx = candidates[trails > self.max_len]
                _mask[fail_idx] = 0
                
        mask[:, -1] = 0 # mask the dummy node for all ants
        go2dummy = (mask[:, :-1] == 0).all(dim=1) # unmask the dummy node for these ants
        mask[go2dummy, -1] = 1
        return mask
    
    def check_done(self, mask):
        # is all masked ?
        return (mask[:, :-1] == 0).all()


N_ITERATIONS = 100
N_ANTS = 20

###################################
# FOR HEURISTIC IMPLEMENTATION


import numpy as np

def heuristics_v2(prize, distance, maxlen):
    n = prize.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        visited = np.zeros(n, dtype=bool)
        remaining_distance = maxlen
        reachable_nodes = [j for j in range(n) if i != j and distance[i][j] <= remaining_distance]

        if reachable_nodes:
            scores = []
            initial_accessibility = (1 - (sum(distance[i][k] for k in reachable_nodes) / (len(reachable_nodes) * maxlen))) if reachable_nodes else 0
            
            for j in reachable_nodes:
                if distance[i][j] > 0:
                    prize_distance_ratio = prize[j] / distance[i][j]
                    adjusted_accessibility = initial_accessibility * (remaining_distance / distance[i][j]) ** 0.5
                    neighborhood_factor = sum(np.exp(-distance[j][k]) for k in reachable_nodes if k != j)
                    combined_score = prize_distance_ratio * adjusted_accessibility * neighborhood_factor
                    scores.append((combined_score, j))

            scores.sort(reverse=True, key=lambda x: x[0])
            current_distance = 0
            decay_factor = 0.9

            for combined_score, j in scores:
                travel_distance = distance[i][j]
                if current_distance + travel_distance <= maxlen:
                    heuristics_matrix[i][j] = combined_score * (decay_factor ** sum(visited)) * (prize[j] ** 1.3)
                    visited[j] = True
                    current_distance += travel_distance
                else:
                    break

    return heuristics_matrix


###################################

from typing import NamedTuple
from torch import Tensor

class OPInstance(NamedTuple):
    n: int
    coordinate: torch.Tensor
    distance: torch.Tensor
    prize: torch.Tensor
    maxlen: float

def gen_prizes(coordinates: Tensor):
    depot_coor = coordinates[0]
    distances = (coordinates - depot_coor).norm(p=2, dim=-1)
    prizes = 1 + torch.floor(99 * distances / distances.max())
    prizes /= prizes.max()
    return prizes

def gen_distance_matrix(coordinates):
    '''
    Args:
        _coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(coordinates)
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9 # note here
    return distances

def get_max_len(n: int) -> float:
    threshold_list = [50, 100, 200, 300]
    maxlen = [3.0,4.0,5.0,6.0]
    for threshold, result in zip(threshold_list, maxlen):
        if n<=threshold:
            return result
    return 7.0

def generate_dataset(filepath, n, batch_size=64):
    coor = np.random.rand(batch_size, n, 2)
    np.savez(filepath, coordinates = coor)

def load_dataset(fp) -> list[OPInstance]:
    data = np.load(fp)
    coordinates = data['coordinates']
    instances = []
    n = coordinates[0].shape[0]
    maxlen = get_max_len(n)
    for coord_np in coordinates:
        coord = torch.from_numpy(coord_np)
        distance = gen_distance_matrix(coord)
        prize = gen_prizes(coord)
        instance = OPInstance(n, coord, distance, prize, maxlen)
        instances.append(instance)
    return instances

#########################################

def solve(inst: OPInstance):
    heu = heuristics(np.array(inst.prize), np.array(inst.distance), inst.maxlen) + 1e-9
    assert tuple(heu.shape) == (inst.n, inst.n)
    heu[heu < 1e-9] = 1e-9
    heu = torch.from_numpy(heu)
    aco = ACO(inst.prize, inst.distance, inst.maxlen, heu, N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return obj

import os, sys

def run(size):
    print(f"Running OP-ACO-EoH for OP{size}...")
    path = os.path.join(os.path.dirname(__file__), 'datasets', f'test_OP{size}.npz')
    dataset = load_dataset(path)
    n_instances = len(dataset)
    LIST = []
    for i, instance in enumerate(dataset):
        costs = solve(instance)
        LIST.append(costs)
    print(np.array(LIST).mean(axis=0).tolist())

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(size)
