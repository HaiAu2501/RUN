import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from F1_final_best import item_compatibility
from F2_final_best import item_badness  
from F3_final_best import insert_position

usecache = True

@nb.njit(nb.int32(nb.int32[:], nb.float32[:], nb.int32), nogil=True, cache=usecache)
def calculate_bins_used(permutation, demands, capacity):
    """
    Calculate number of bins used with automatic bin switching.
    Processes permutation sequentially, switching to new bin when capacity exceeded.
    """
    if len(permutation) == 0:
        return 0
    
    bins_used = 1
    current_load = np.float32(0.0)
    
    for i in range(len(permutation)):
        item = permutation[i]
        item_size = demands[item]
        
        # Check if adding this item exceeds capacity
        if current_load + item_size > capacity:
            # Start new bin
            bins_used += 1
            current_load = item_size
        else:
            # Add to current bin
            current_load += item_size
    
    return bins_used

@nb.njit(nb.int8(nb.int32[:], nb.int32), nogil=True, cache=usecache)
def is_valid_permutation_bpp(permutation, n_items):
    """Check if permutation is valid (visits all items exactly once)."""
    if len(permutation) != n_items:
        return False
    
    visited = np.zeros(n_items, dtype=nb.int8)
    
    for i in range(len(permutation)):
        item = permutation[i]
        if item < 0 or item >= n_items or visited[item]:
            return False
        visited[item] = 1
    
    return True

@nb.njit(nb.int32[:](nb.float32[:, :], nb.float32[:], nb.int32, nb.int32), nogil=True, cache=usecache)
def greedy_construction_bpp(compatibility_scores, demands, capacity, start_item):
    """Construct BPP permutation greedily using F1 compatibility scores."""
    n_items = compatibility_scores.shape[0]
    
    permutation = np.zeros(n_items, dtype=nb.int32)
    visited = np.zeros(n_items, dtype=nb.int8)
    
    # Start from given item
    permutation[0] = start_item
    visited[start_item] = 1
    current_item = start_item
    
    # Greedy construction
    for step in range(1, n_items):
        best_item = -1
        best_score = np.float32(-np.inf)
        
        for next_item in range(n_items):
            if visited[next_item] == 0:
                score = compatibility_scores[current_item, next_item]
                if score > best_score:
                    best_score = score
                    best_item = next_item
        
        # Fallback if no good item found
        if best_item == -1:
            for next_item in range(n_items):
                if visited[next_item] == 0:
                    best_item = next_item
                    break
        
        if best_item != -1:
            permutation[step] = best_item
            visited[best_item] = 1
            current_item = best_item
    
    return permutation

def precompute_compatibility_scores(demands, capacity):
    """Precompute all item compatibility scores using F1."""
    n_items = len(demands)
    compatibility_scores = np.full((n_items, n_items), -np.inf, dtype=np.float32)
    
    for i in range(n_items):
        for j in range(n_items):
            if i != j:
                compatibility_scores[i, j] = item_compatibility(i, j, demands, capacity)
            else:
                compatibility_scores[i, j] = 0.0  # Neutral score for same item
    
    return compatibility_scores

def deconstruct_permutation_bpp(permutation_array, demands, capacity, destruction_rate=0.3):
    """Deconstruct BPP permutation by removing bad items using F2."""
    n = len(permutation_array)
    if n == 0:
        return np.array([], dtype=np.int32), permutation_array.copy()
    
    # Calculate badness scores using F2
    badness_scores = np.zeros(n, dtype=np.float32)
    permutation_list = permutation_array.tolist()
    
    for i in range(n):
        badness_scores[i] = item_badness(i, permutation_list, demands, capacity)
    
    # Determine items to remove
    num_to_remove = max(1, int(n * destruction_rate))
    num_to_remove = min(num_to_remove, n - 2)  # Keep at least 2 items
    
    if num_to_remove >= n:
        return permutation_array.copy(), np.array([], dtype=np.int32)
    
    # Get indices of worst items
    worst_indices = np.argpartition(badness_scores, -num_to_remove)[-num_to_remove:]
    
    # Create removal mask
    removal_mask = np.zeros(n, dtype=bool)
    removal_mask[worst_indices] = True
    
    # Split permutation
    removed_items = permutation_array[removal_mask].copy()
    remaining_permutation = permutation_array[~removal_mask].copy()
    
    return removed_items, remaining_permutation

def repair_permutation_bpp(removed_items, partial_permutation, demands, capacity):
    """Repair BPP permutation by inserting removed items using F3."""
    if len(removed_items) == 0:
        return partial_permutation.copy()
    
    current_permutation = partial_permutation.tolist()
    
    # Sort removed items by size (smaller items first - easier to place)
    item_sizes = [(demands[item], item) for item in removed_items]
    item_sizes.sort()  # Sort by size (smaller first)
    sorted_removed = [item for _, item in item_sizes]
    
    # Insert each removed item at best position using F3
    for item in sorted_removed:
        if len(current_permutation) == 0:
            current_permutation.append(int(item))
        else:
            position = insert_position(int(item), current_permutation, demands, capacity)
            current_permutation.insert(position, int(item))
    
    return np.array(current_permutation, dtype=np.int32)

def single_dr_run_bpp(demands, capacity, start_item, destruction_rate=0.3):
    """Single Deconstruction-Repair run for BPP from given start item."""
    try:
        demands_float32 = demands.astype(np.float32)
        n_items = len(demands)
        
        # 1. Precompute compatibility scores
        compatibility_scores = precompute_compatibility_scores(demands, capacity)
        compatibility_scores_float32 = compatibility_scores.astype(np.float32)
        
        # 2. Greedy construction using F1
        permutation = greedy_construction_bpp(compatibility_scores_float32, demands_float32, capacity, start_item)
        
        # Validate initial permutation
        if not is_valid_permutation_bpp(permutation, n_items):
            # Fallback: simple sequential permutation
            permutation = np.arange(n_items, dtype=np.int32)
            np.random.shuffle(permutation)
        
        # 3. Deconstruction using F2
        removed_items, partial_permutation = deconstruct_permutation_bpp(
            permutation, demands, capacity, destruction_rate)
        
        # 4. Repair using F3
        repaired_permutation = repair_permutation_bpp(
            removed_items, partial_permutation, demands, capacity)
        
        # Validate repaired permutation
        if not is_valid_permutation_bpp(repaired_permutation, n_items):
            repaired_permutation = permutation
        
        # Calculate final bins used
        bins_used = int(calculate_bins_used(repaired_permutation, demands_float32, capacity))
        
        return repaired_permutation, bins_used
        
    except Exception as e:
        # Fallback: return simple permutation
        fallback_permutation = np.arange(n_items, dtype=np.int32)
        fallback_bins = int(calculate_bins_used(fallback_permutation, demands.astype(np.float32), capacity))
        return fallback_permutation, fallback_bins

def run_bpp_dr(demands: np.ndarray, 
               capacity: int,
               destruction_rate: float = 0.3,
               max_workers: int = None,
               seed: int = None) -> int:
    """
    Deconstruction-Repair algorithm for BPP with parallel execution.
    
    Args:
        demands: Item sizes array (n,)
        capacity: Bin capacity constraint
        destruction_rate: Fraction of items to remove (0.1 to 0.5)
        max_workers: Maximum number of threads (None = n_items)
        seed: Random seed for reproducibility
    
    Returns:
        Minimum number of bins used
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    n_items = len(demands)
    
    # Handle edge cases
    if n_items == 0:
        return 0
    if n_items == 1:
        return 1
    
    # Determine number of workers
    if max_workers is None:
        max_workers = n_items
    else:
        max_workers = min(max_workers, n_items)
    
    best_permutation = None
    best_bins = n_items  # Worst case: each item in separate bin
    
    # Parallel execution: try all starting items
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(single_dr_run_bpp, demands, capacity, start_item, destruction_rate)
            for start_item in range(n_items)
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                permutation, bins_used = future.result()
                if bins_used < best_bins:
                    best_bins = bins_used
                    best_permutation = permutation
            except Exception as e:
                continue
    
    # Validate final result
    if best_permutation is None or best_bins >= n_items:
        # Ultimate fallback
        fallback_permutation = np.arange(n_items, dtype=np.int32)
        best_bins = int(calculate_bins_used(fallback_permutation, demands.astype(np.float32), capacity))
    
    return best_bins