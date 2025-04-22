import os
import numpy as np

def generate_large_scale_instances():
    basepath = os.path.join(os.path.dirname(__file__), "ls_tsp")
    os.makedirs(basepath, exist_ok=True)

    np.random.seed(42)

    for size in [200, 500, 1000]:
        n_instances = 128
        dataset = np.random.rand(n_instances, size, 2)
        np.save(os.path.join(basepath, f"TSP{size}.npy"), dataset)

if __name__ == "__main__":
    generate_large_scale_instances()