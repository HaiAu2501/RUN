import os
import numpy as np
from numpy.random import beta

def generate_new_instances():
    basepath = os.path.join(os.path.dirname(__file__), "beta_tsp")
    os.makedirs(basepath, exist_ok=True)

    np.random.seed(42)

    for size in [20, 50, 100, 200, 500, 1000]:
        n_instances = 128
        # Sinh theo phân phối Beta lệch về phía 0 (a=2, b=5)
        x_coords = beta(a=2, b=5, size=(n_instances, size))
        y_coords = beta(a=5, b=2, size=(n_instances, size))  # lệch về phía 1
        dataset = np.stack([x_coords, y_coords], axis=-1)  # shape: (n_instances, size, 2)

        np.save(os.path.join(basepath, f"TSP{size}.npy"), dataset)

def generate_normal_instances():
    basepath = os.path.join(os.path.dirname(__file__), "normal_tsp")
    os.makedirs(basepath, exist_ok=True)

    np.random.seed(42)

    for size in [20, 50, 100, 200, 500, 1000]:
        n_instances = 128
        # Sinh tọa độ theo phân phối chuẩn, sau đó ép vào [0, 1]
        coords = np.random.normal(loc=0.5, scale=0.15, size=(n_instances, size, 2))
        coords = np.clip(coords, 0, 1)

        np.save(os.path.join(basepath, f"TSP{size}.npy"), coords)

if __name__ == "__main__":
    generate_new_instances()
    generate_normal_instances()