import numpy as np
import os

def generate_tsp_datasets():
    np.random.seed(1234)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_dir = os.path.join(current_dir, "datasets")

    os.makedirs(dataset_dir, exist_ok=True)
    
    # Train
    for size in [50]:
        n_instances = 5
        batch = np.random.rand(n_instances, size, 2)
        filename = os.path.join(dataset_dir, f"train_TSP{size}.npy")
        np.save(filename, batch)
        print(f"Generated {n_instances} training instances of size {size}")

    # Validation
    for size in [20, 50, 100]:
        n_instances = 64
        batch = np.random.rand(n_instances, size, 2)
        filename = os.path.join(dataset_dir, f"val_TSP{size}.npy")
        np.save(filename, batch)
        print(f"Generated {n_instances} validation instances of size {size}")

    # Test
    for size in [20, 50, 100, 200, 500, 1000]:
        n_instances = 64
        batch = np.random.rand(n_instances, size, 2)
        filename = os.path.join(dataset_dir, f"test_TSP{size}.npy")
        np.save(filename, batch)
        print(f"Generated {n_instances} test instances of size {size}")

if __name__ == "__main__":
    generate_tsp_datasets()