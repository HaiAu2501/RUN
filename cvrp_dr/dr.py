import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from F1_final_best import edge_score
from F2_final_best import customer_badness  
from F3_final_best import insert_position

usecache = True

@nb.njit(nb.float32(nb.int32[:], nb.float32[:, :], nb.float32[:], nb.int32), nogil=True, cache=usecache)
def calculate_cvrp_cost(permutation, distances, demands, capacity):
    """
    Calculate total CVRP cost with automatic route splitting.
    Note: Node 0 is the depot, all routes start and end at depot.
    """
    total_cost = np.float32(0.0)
    current_load = np.float32(0.0)
    current_pos = 0  # Start at depot
    
    for i in range(len(permutation)):
        customer = permutation[i]
        
        # Check if adding this customer exceeds capacity
        if current_load + demands[customer] > capacity:
            # Return to depot and start new route
            total_cost += distances[current_pos, 0]  # Return to depot
            current_pos = 0  # Now at depot
            current_load = np.float32(0.0)
        
        # Go from current position to customer
        total_cost += distances[current_pos, customer]
        current_pos = customer
        current_load += demands[customer]
    
    # Final return to depot
    total_cost += distances[current_pos, 0]
    return total_cost

@nb.njit(nb.int8(nb.int32[:], nb.int32), nogil=True, cache=usecache)
def is_valid_permutation(permutation, n_customers):
    """Check if permutation is valid (excludes depot, visits all customers)."""
    if len(permutation) != n_customers - 1:  # Exclude depot
        return False
    
    visited = np.zeros(n_customers, dtype=nb.int8)
    visited[0] = 1  # Depot is always "visited"
    
    for i in range(len(permutation)):
        customer = permutation[i]
        if customer <= 0 or customer >= n_customers or visited[customer]:
            return False
        visited[customer] = 1
    
    return True

@nb.njit(nb.int32[:](nb.float32[:, :], nb.float32[:], nb.int32, nb.int32), nogil=True, cache=usecache)
def greedy_construction_cvrp(edge_scores, demands, capacity, start_customer):
    """Construct CVRP permutation greedily using F1 edge scores."""
    n_customers = edge_scores.shape[0]
    
    permutation = np.zeros(n_customers - 1, dtype=nb.int32)  # Exclude depot (node 0)
    visited = np.zeros(n_customers, dtype=nb.int8)
    visited[0] = 1  # Depot is always visited
    
    # Start from given customer
    permutation[0] = start_customer
    visited[start_customer] = 1
    current_customer = start_customer
    current_load = demands[start_customer]
    
    # Greedy construction
    for step in range(1, n_customers - 1):
        best_customer = -1
        best_score = np.float32(-np.inf)
        
        for next_customer in range(1, n_customers):  # Skip depot (0)
            if visited[next_customer] == 0:
                # Check capacity constraint
                if current_load + demands[next_customer] <= capacity:
                    score = edge_scores[current_customer, next_customer]
                    if score > best_score:
                        best_score = score
                        best_customer = next_customer
        
        # If no feasible customer found, pick any unvisited one
        if best_customer == -1:
            for next_customer in range(1, n_customers):
                if visited[next_customer] == 0:
                    best_customer = next_customer
                    break
        
        if best_customer != -1:
            permutation[step] = best_customer
            visited[best_customer] = 1
            
            # Update load (with automatic reset logic handled in cost calculation)
            if current_load + demands[best_customer] > capacity:
                current_load = demands[best_customer]  # Start new route
            else:
                current_load += demands[best_customer]
            
            current_customer = best_customer
    
    return permutation

@nb.njit(nb.void(nb.int32[:], nb.float32[:, :], nb.float32[:], nb.int32, nb.int32), nogil=True, cache=usecache)
def two_opt_cvrp(permutation, distances, demands, capacity, max_iterations=30):
    """2-opt improvement for CVRP with capacity constraints."""
    n = len(permutation)
    if n < 4:
        return
    
    iteration = 0
    while iteration < max_iterations:
        improved = False
        best_improvement = np.float32(0.0)
        best_i = -1
        best_j = -1
        
        # Try all 2-opt moves
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j >= n:
                    break
                
                # Calculate current cost
                old_cost = calculate_cvrp_cost(permutation, distances, demands, capacity)
                
                # Apply 2-opt move (reverse segment [i+1:j+1])
                left = i + 1
                right = j
                while left < right:
                    permutation[left], permutation[right] = permutation[right], permutation[left]
                    left += 1
                    right -= 1
                
                # Calculate new cost
                new_cost = calculate_cvrp_cost(permutation, distances, demands, capacity)
                improvement = old_cost - new_cost
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i = i
                    best_j = j
                else:
                    # Revert the move
                    left = i + 1
                    right = j
                    while left < right:
                        permutation[left], permutation[right] = permutation[right], permutation[left]
                        left += 1
                        right -= 1
        
        # Apply best improvement if found
        if best_improvement > 1e-6:
            left = best_i + 1
            right = best_j
            while left < right:
                permutation[left], permutation[right] = permutation[right], permutation[left]
                left += 1
                right -= 1
            improved = True
        
        if not improved:
            break
            
        iteration += 1

def precompute_edge_scores_cvrp(distances, demands, capacity):
    """Precompute all edge scores for CVRP using F1."""
    n_customers = distances.shape[0]
    edge_scores = np.full((n_customers, n_customers), -np.inf, dtype=np.float32)
    
    for i in range(n_customers):
        for j in range(n_customers):
            if i != j:
                # Only compute scores for edges that don't involve depot as destination
                # (since we handle depot transitions automatically in cost calculation)
                if j > 0:  # j is a customer, not depot
                    edge_scores[i, j] = edge_score(i, j, distances, demands, capacity)
                else:  # j is depot, give neutral score
                    edge_scores[i, j] = 0.0
    
    return edge_scores

def deconstruct_permutation_cvrp(permutation_array, distances, demands, capacity, destruction_rate=0.3):
    """Deconstruct CVRP permutation by removing bad customers using F2."""
    n = len(permutation_array)
    if n == 0:
        return np.array([], dtype=np.int32), permutation_array.copy()
    
    # Calculate badness scores using F2
    badness_scores = np.zeros(n, dtype=np.float32)
    permutation_list = permutation_array.tolist()
    
    for i in range(n):
        badness_scores[i] = customer_badness(i, permutation_list, distances, demands, capacity)
    
    # Determine customers to remove
    num_to_remove = max(1, int(n * destruction_rate))
    num_to_remove = min(num_to_remove, n - 2)  # Keep at least 2 customers
    
    if num_to_remove >= n:
        return permutation_array.copy(), np.array([], dtype=np.int32)
    
    # Get indices of worst customers
    worst_indices = np.argpartition(badness_scores, -num_to_remove)[-num_to_remove:]
    
    # Create removal mask
    removal_mask = np.zeros(n, dtype=bool)
    removal_mask[worst_indices] = True
    
    # Split permutation
    removed_customers = permutation_array[removal_mask].copy()
    remaining_permutation = permutation_array[~removal_mask].copy()
    
    return removed_customers, remaining_permutation

def repair_permutation_cvrp(removed_customers, partial_permutation, distances, demands, capacity):
    """Repair CVRP permutation by inserting removed customers using F3."""
    if len(removed_customers) == 0:
        return partial_permutation.copy()
    
    current_permutation = partial_permutation.tolist()
    
    # Sort removed customers by distance to current permutation (nearest first)
    if len(current_permutation) > 0:
        customer_distances = []
        for customer in removed_customers:
            min_dist = min(distances[customer, perm_customer] for perm_customer in current_permutation)
            customer_distances.append((min_dist, customer))
        
        customer_distances.sort()
        sorted_removed = [customer for _, customer in customer_distances]
    else:
        sorted_removed = removed_customers.tolist()
    
    # Insert each removed customer at best position using F3
    for customer in sorted_removed:
        if len(current_permutation) == 0:
            current_permutation.append(int(customer))
        else:
            position = insert_position(int(customer), current_permutation, distances, demands, capacity)
            current_permutation.insert(position, int(customer))
    
    return np.array(current_permutation, dtype=np.int32)

def single_dr_run_cvrp(distances, demands, capacity, start_customer, destruction_rate=0.3, use_2opt=True):
    """Single Deconstruction-Repair run for CVRP from given start customer."""
    try:
        distances_float32 = distances.astype(np.float32)
        demands_float32 = demands.astype(np.float32)
        n_customers = distances.shape[0]
        
        # 1. Precompute edge scores
        edge_scores = precompute_edge_scores_cvrp(distances, demands, capacity)
        edge_scores_float32 = edge_scores.astype(np.float32)
        
        # 2. Greedy construction using F1
        permutation = greedy_construction_cvrp(edge_scores_float32, demands_float32, capacity, start_customer)
        
        # Validate initial permutation
        if not is_valid_permutation(permutation, n_customers):
            # Fallback: simple sequential permutation
            permutation = np.arange(1, n_customers, dtype=np.int32)
            np.random.shuffle(permutation)
        
        # 3. Deconstruction using F2
        removed_customers, partial_permutation = deconstruct_permutation_cvrp(
            permutation, distances, demands, capacity, destruction_rate)
        
        # 4. Repair using F3
        repaired_permutation = repair_permutation_cvrp(
            removed_customers, partial_permutation, distances, demands, capacity)
        
        # Validate repaired permutation
        if not is_valid_permutation(repaired_permutation, n_customers):
            repaired_permutation = permutation
        
        # 5. Optional 2-opt improvement
        if use_2opt and len(repaired_permutation) > 3:
            two_opt_cvrp(repaired_permutation, distances_float32, demands_float32, capacity, max_iterations=10)
        
        # Calculate final cost
        final_cost = float(calculate_cvrp_cost(repaired_permutation, distances_float32, demands_float32, capacity))
        
        return repaired_permutation, final_cost
        
    except Exception as e:
        # Fallback: return simple permutation
        fallback_permutation = np.arange(1, n_customers, dtype=np.int32)
        fallback_cost = float(calculate_cvrp_cost(fallback_permutation, distances.astype(np.float32), 
                                                demands.astype(np.float32), capacity))
        return fallback_permutation, fallback_cost

def run_cvrp_dr(distances: np.ndarray, 
                demands: np.ndarray,
                capacity: int,
                destruction_rate: float = 0.3,
                use_2opt: bool = True,
                max_workers: int = None,
                seed: int = None) -> float:
    """
    Deconstruction-Repair algorithm for CVRP with parallel execution.
    Note: Node 0 is the depot, customers are nodes 1 to n-1.
    
    Args:
        distances: Distance matrix (n x n) with depot at index 0
        demands: Demand array (n,) with depot demand = 0
        capacity: Vehicle capacity constraint
        destruction_rate: Fraction of customers to remove (0.1 to 0.5)
        use_2opt: Whether to apply 2-opt local improvement
        max_workers: Maximum number of threads (None = n_customers-1)
        seed: Random seed for reproducibility
    
    Returns:
        Best total cost found
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    n_customers = distances.shape[0]
    
    # Determine number of workers (exclude depot)
    n_actual_customers = n_customers - 1
    if max_workers is None:
        max_workers = n_actual_customers
    else:
        max_workers = min(max_workers, n_actual_customers)
    
    best_permutation = None
    best_cost = float('inf')
    
    # Parallel execution: try all starting customers (excluding depot)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(single_dr_run_cvrp, distances, demands, capacity, 
                          start_customer, destruction_rate, use_2opt)
            for start_customer in range(1, n_customers)  # Exclude depot (0)
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                permutation, cost = future.result()
                if cost < best_cost:
                    best_cost = cost
                    best_permutation = permutation
            except Exception as e:
                continue
    
    # Validate final result
    if best_permutation is None or best_cost == float('inf'):
        # Ultimate fallback
        fallback_permutation = np.arange(1, n_customers, dtype=np.int32)
        best_cost = float(calculate_cvrp_cost(fallback_permutation, distances.astype(np.float32), 
                                            demands.astype(np.float32), capacity))
    
    return best_cost