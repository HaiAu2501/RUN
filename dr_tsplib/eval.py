import tsplib95
import numpy as np
import os, sys
from copy import copy
from dr import run_tsp_dr

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

def solve(name):
    problem = tsplib95.load(f'{ben_path}/{name}.tsp')
    n = len(list(problem.get_nodes()))
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
                dist_mat[i, j] = problem.get_weight(i + 1, j + 1)
                dist_mat[j, i] = dist_mat[i, j]

    total_distance = run_tsp_dr(dist_mat, use_2opt=True)

    opt_gap = (total_distance - OPTIMAL[name]) / OPTIMAL[name] * 100
    print(f"{name}: opt gap = {opt_gap:.2f}%")

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "tsp225"
    solve(name)