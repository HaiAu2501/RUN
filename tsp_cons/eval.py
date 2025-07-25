import os, sys
import tsplib95
import numpy as np
from copy import copy
from scipy.spatial import distance_matrix
from F1_final_best import select_next_node

PROBLEMS = [
    "tsp225",
    "rat99",
    "bier127",
    "lin318",
    "eil51",
    "d493",
    "kroB100",
    "kroC100",
    "ch130",
    "pr299",
    "fl417",
    "kroA150",
    "pr264",
    "pr226",
    "pr439"
]

OPTIMAL = {}

cur_dir = os.path.dirname(os.path.abspath(__file__))
sol_path = os.path.join(cur_dir, "solutions")
ben_path = os.path.join(cur_dir, "benchmark")

with open(sol_path, "r", encoding="utf-8") as f:
    for line in f:
        name, opt = line.strip().split(':')
        OPTIMAL[name.strip()] = int(opt.strip())

def solve():
    for name in PROBLEMS:
        problem = tsplib95.load(f'{ben_path}/{name}.tsp')
        n = len(list(problem.get_nodes()))
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                    dist_mat[i, j] = problem.get_weight(i + 1, j + 1)
                    dist_mat[j, i] = dist_mat[i, j]

        start = 0
        solution = [start]
        unvisited = list(range(1, n))

        for _ in range(n - 1):
            next_node = select_next_node(solution[-1], 0, copy(unvisited), copy(dist_mat))
            assert next_node in unvisited, "Next node must be unvisited"
            solution.append(next_node)
            unvisited.remove(next_node)

        solution.append(0)  # Return to the starting node
        total_distance = sum(dist_mat[solution[i], solution[i + 1]] for i in range(n))
        opt_gap = (total_distance - OPTIMAL[name]) / OPTIMAL[name] * 100
        print(f"{name}: opt gap = {opt_gap:.2f}%")

if __name__ == "__main__":
    solve()