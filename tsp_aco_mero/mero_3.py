from mero import *
import numpy as np
import torch
import tsplib95
import os
from typing import List, Dict, Any, Optional, List
from abc import ABC, abstractmethod






def run_aco(n_ants=30, n_iterations=100):
    # Lấy tất cả các file trong thư mục benchmark
    for file in os.listdir('benchmark'):
        if file.endswith('.tsp'):
            # Lấy tên file mà không có phần mở rộng
            name = os.path.splitext(file)[0]
            # Đọc dữ liệu từ file
            distances, optimal = get_data(name)
            def opt_gap(optimal, obj):
                return (obj - optimal) / optimal * 100

            # Chạy hàm solve_reevo với dữ liệu đã đọc
            avg_obj = 0
            for seed in range(5):
                heuristic_strategy = HeuristicImpl()
                probability_strategy = ProbabilityImpl()
                pheromone_strategy = PheromoneImpl()
                
                aco = AntColonyOptimization(
                    distances=distances,
                    n_ants=n_ants,
                    n_iterations=n_iterations,
                    device='cpu',
                    seed=seed,
                    heuristic_strategy=heuristic_strategy,
                    probability_strategy=probability_strategy,
                    pheromone_strategy=pheromone_strategy
                )
                
                obj = aco.run()
                avg_obj += obj

            avg_obj /= 5
            print(f"{name}: opt_gap = {opt_gap(optimal, avg_obj):.2f}%")

if __name__ == "__main__":
    run_aco(n_ants=30, n_iterations=100)