import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Annotated
from math import floor

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float_]

def organize_path(path: IntArray) -> Tuple[int, IntArray]:
    order = {}
    result = np.zeros_like(path)
    for i, v in enumerate(path):
        if v in order:
            result[i] = order[v]
        else:
            result[i] = order[v] = len(order)
    return len(order), result

def calculate_path_cost_fitness(vacancies: IntArray, capacity: int) -> Tuple[int, float]:
    occupied = (capacity - vacancies[vacancies!=capacity]).astype(float)
    cost = len(occupied)
    result = ((occupied/capacity)**2).sum().item()/cost
    return cost, result

def calculate_path_fitness(vacancies: List[int], capacity: int) -> float:
    occupied = capacity - np.array(vacancies, dtype=float)
    result = ((occupied/capacity)**2).sum().item()/len(vacancies)
    return result

def greedy_sample(prob: FloatArray) -> int:
    return prob.argmax().item()

def random_sample(prob: FloatArray) -> int:
    # not used, `random_sample_discrete_distribution` is a faster implementation
    sampled = np.random.choice(prob.size, p=prob/prob.sum())
    return sampled

def random_sample_discrete_distribution(prob: FloatArray) -> int:
    # prob_exp = np.exp(prob-prob.max())
    # prob_exp[prob==0] = 0
    # np.random.choice is somehow slow
    cumprob = np.cumsum(prob)
    sampled = np.searchsorted(cumprob, next(uniform_generator)*cumprob[-1]).item()
    return sampled if sampled<len(cumprob) else len(cumprob)-1

def uniform_number_generator(batch_size = 500):
    # it's also slow to generate random numbers one by one
    while 1:
        numbers = np.random.random(batch_size)
        for n in numbers:
            yield n.item()

uniform_generator = uniform_number_generator()

class ACO(object):
    def __init__(self,
                 demand: IntArray,   # (n, )
                 heuristic: FloatArray,   # (n, n)
                 capacity: int,
                 n_ants=20, 
                 decay=0.95,
                 alpha=1,
                 beta=1,
                 greedy = False,
                 seed=0
                 ):
        
        self.problem_size = len(demand)
        self.capacity = capacity
        self.demand = demand
        assert self.demand.max() <= self.capacity
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone: FloatArray = np.ones((self.problem_size, self.problem_size)) # problem_size x self.problem_size
        heuristic[heuristic > 1e6] = 1e6
        heuristic[heuristic < 1e-6] = 1e-6
        heuristic = heuristic/heuristic.max() # normalize
        heuristic[heuristic < 1e-6] = 1e-6
        self.heuristic: FloatArray = heuristic # problem_size x self.problem_size

        self.shortest_path: IntArray = np.arange(self.problem_size)
        self.best_cost = self.problem_size

        self._ordinal: IntArray = np.arange(self.problem_size, dtype=int) # for indexing
        self.greedy_mode = greedy
        self.seed = seed
    
    def run(self, iterations: int) -> Tuple[int, IntArray]:
        list_obj = []
        np.random.seed(self.seed)
        for it in range(iterations + 1):
            prob = self.pheromone**self.alpha * self.heuristic**self.beta
            paths, costs, fitnesses = self.gen_paths(self.n_ants, prob)
            best_index = costs.argmin()
            best_cost = costs[best_index].item()
            if best_cost < self.best_cost:
                self.shortest_path = paths[best_index]
                self.best_cost = best_cost
            self.update_pheronome(paths, fitnesses)
            if it % 5 == 0:
                list_obj.append(self.best_cost)
        assert self.is_valid_path(self.shortest_path)
        # cost, path = organize_path(self.shortest_path)
        # assert cost >= np.ceil(np.sum(self.demand).astype(float)/self.capacity).item()
        cost, _ = organize_path(self.shortest_path)
        return list_obj

    def sample_only(self, count: int) -> Tuple[int, IntArray]:
        self.greedy_mode = True
        paths, costs, _ = self.gen_paths(count, self.heuristic)
        best_index = costs.argmin()
        best_path = paths[best_index]
        assert self.is_valid_path(best_path)
        return organize_path(best_path)

    def update_pheronome(self, paths: List[IntArray], fitnesses: FloatArray):
        delta_phe = np.zeros_like(self.pheromone) # problem_size x problem_size
        for path, f in zip(paths, fitnesses):
            delta_phe[path[:, None]==path[None, :]] += f / self.n_ants
        self.pheromone *= self.decay
        self.pheromone += delta_phe

    def gen_paths(self, count: int, prob: FloatArray) -> Tuple[List[IntArray], IntArray, FloatArray]:
        paths, costs, fitnesses = [], [], []
        for _ in range(count):
            path, cost, fitness = self.sample_path(prob)
            paths.append(path)
            costs.append(cost)
            fitnesses.append(fitness)
        return paths, np.array(costs, dtype=int), np.array(fitnesses, dtype=float)
    
    def sample_path(self, prob: FloatArray
                    ) -> Tuple[
                        Annotated[IntArray, "sampled path"], 
                        Annotated[int, "used bins"], 
                        Annotated[float, "fitness"]]:
        
        if self.greedy_mode:
            sample_func = greedy_sample
        else:
            sample_func = random_sample_discrete_distribution
    
        path = np.ones(self.problem_size, dtype=int)*-1 # x=path[i] => put item i in bin x
        valid_items = np.ones(self.problem_size, dtype=bool)
        current_bin = item_count = 0
        vacancies = []
        bin_vacancy = self.capacity
        bin_items = np.zeros_like(valid_items)

        for _ in range(self.problem_size):
            mask = np.bitwise_and(self.demand <= bin_vacancy, valid_items)
            if not np.any(mask): # no valid item
                # move to the next bin
                vacancies.append(bin_vacancy)
                bin_vacancy, item_count = self.capacity, 0
                current_bin += 1
                bin_items[:] = False
                # uniformly select one
                selected = self.random_select(valid_items)
            else:
                if item_count == 0:
                    selected = self.random_select(mask)
                else:
                    item_prob = (prob[bin_items].sum(0)/item_count+1e-5) * mask
                    selected = sample_func(item_prob)
            
            # put item in this bin
            bin_items[selected] = True
            bin_vacancy -= self.demand[selected]
            valid_items[selected] = False
            path[selected] = current_bin
            item_count += 1
        
        vacancies.append(bin_vacancy)
        fitness = calculate_path_fitness(vacancies, self.capacity)
        return path, len(vacancies), fitness
    
    def random_select(self, mask: npt.NDArray[np.bool_]) -> int:
        valid = self._ordinal[mask]
        return valid[floor(next(uniform_generator)*len(valid))].item()
        # return valid[np.random.randint(0, len(valid))].item()
    
    def is_valid_path(self, path: IntArray) -> bool:
        # not used
        if path.shape[0] != self.problem_size:
            return False
        bins, path = organize_path(path)
        occupied = np.zeros(bins, dtype=int)
        for i, v in enumerate(path):
            if v<0:
                return False
            occupied[v] += self.demand[i]
            if occupied[v] > self.capacity:
                return False
        return True

N_ITERATIONS = 50
N_ANTS = 20
SAMPLE_COUNT = 200

########## HEURISTICS ##########


import numpy as np

def heuristics(demand, capacity):
    n = len(demand)
    heuristics_matrix = np.full((n, n), -1)  # Start with all pairs as unsuitable

    # Sort indices of items based on demand in ascending order for different pairing strategy
    sorted_indices = np.argsort(demand)

    # Calculate frequency of each item size to weight pairs
    size_frequency = {size: list(demand).count(size) for size in set(demand)}

    for i in range(n):
        for j in range(i + 1, n):
            total_size = demand[sorted_indices[i]] + demand[sorted_indices[j]]
            if total_size <= capacity:
                available_space = capacity - total_size

                # Calculate size correlation factor with a modified log component
                size_difference = abs(demand[sorted_indices[i]] - demand[sorted_indices[j]])
                size_correlation = 1 - (size_difference / (np.log(max(demand[sorted_indices[i]], demand[sorted_indices[j]], 1) + 1))) ** 2

                # Quadratic penalty for larger pairs based on size frequency
                penalty = (size_frequency[demand[sorted_indices[i]]] + size_frequency[demand[sorted_indices[j]]]) ** 2 * 0.05

                # Combined score with new factor adjustments
                combined_score = (capacity / (available_space + 1e-6)) ** 2 + (size_correlation * 2) - (total_size * 0.5) - penalty  # Adjusted factors
                heuristics_matrix[sorted_indices[i]][sorted_indices[j]] = combined_score
                heuristics_matrix[sorted_indices[j]][sorted_indices[i]] = combined_score  # Symmetric matrix

    return heuristics_matrix



import os, sys


def run(size):
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, "datasets", f"test_BPP{size}.npz")
    dataset = np.load(dataset_path)
    inst = dataset['instances']
    n_inst = inst.shape[0]
    list_obj = []
    for n in range(n_inst):
        cap = int(inst[n][0])
        demands = inst[n][1:].astype(int)
        heu = heuristics(demands.copy(), cap)
        aco = ACO(demands, heu.astype(float), capacity=cap, n_ants=N_ANTS, greedy=False)
        obj = aco.run(N_ITERATIONS)
        list_obj.append(obj)
    print(np.array(list_obj).mean(axis=0).tolist())

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(size)
