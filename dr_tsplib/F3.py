# Final round optimized implementation for insert_position
# Strategy ID: F3
# Phase: Final round (system-aware)

import numpy as np

def insert_position(city: int, incomplete_tour: list[int], distances: np.ndarray) -> int:
    if not incomplete_tour:  # Handle empty tour
        return 0
    if len(incomplete_tour) == 1:  # Handle single city tour
        return 1

    best_position = 0
    min_increase = float('inf')
    n = len(incomplete_tour)
    cache = np.zeros(n + 1)  # Cache for evaluated insertion costs
    average_distance = np.mean(distances)  # Compute average distance

    for i in range(n + 1):  # Check all potential insertion points
        if i == 0:
            cost_increase = distances[city, incomplete_tour[0]]
        elif i == n:
            cost_increase = (distances[incomplete_tour[-1], city] + distances[city, incomplete_tour[0]] - distances[incomplete_tour[-1], incomplete_tour[0]])
        else:
            cost_increase = (distances[incomplete_tour[i - 1], city] + distances[city, incomplete_tour[i]] - distances[incomplete_tour[i - 1], incomplete_tour[i]])

        # Optionally skip positions deemed less significant
        if cost_increase > average_distance:
            continue

        # Compute angular penalty more dynamically based on spatial relations
        if i > 0 and i < n:
            prev = incomplete_tour[i - 1]
            next_city = incomplete_tour[i]
            angle_penalty = np.abs(np.arctan2((distances[prev, next_city] + distances[prev, city]), distances[city, next_city]) - 
                                        np.arctan2(distances[prev, next_city], distances[prev, city]))
            cost_increase *= (1 + angle_penalty / np.pi)  # Normalize to pi range

        # Use cache for previously computed minimum costs
        if cache[i] == 0:
            cache[i] = cost_increase
        else:
            cost_increase = cache[i]

        if cost_increase < min_increase:
            min_increase = cost_increase
            best_position = i

    return best_position