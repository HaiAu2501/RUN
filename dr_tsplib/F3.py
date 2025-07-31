# Final round optimized implementation for insert_position
# Strategy ID: F3
# Phase: Final round (system-aware)

import numpy as np

def insert_position(city: int, incomplete_tour: list[int], distances: np.ndarray) -> int:
    n = len(incomplete_tour)
    if n == 0:
        return 0  # Empty tour
    if n == 1:
        return 1  # One city in tour

    # Precompute current total distance of the incomplete tour
    total_distance = 0
    for i in range(n - 1):
        total_distance += distances[incomplete_tour[i], incomplete_tour[i + 1]]

    min_increase = float('inf')
    best_position = 0

    # Evaluate cost of insertion at all positions
    for i in range(n + 1):
        if i == 0:
            cost_increase = distances[city, incomplete_tour[0]]  # Before the first city
        elif i == n:
            cost_increase = distances[incomplete_tour[-1], city]   # After the last city
        else:
            prev_city = incomplete_tour[i - 1]
            next_city = incomplete_tour[i]
            cost_increase = (distances[prev_city, city] + distances[city, next_city] - 
                             distances[prev_city, next_city])  # In the middle

        # Update best position if current is better
        if cost_increase < min_increase:
            min_increase = cost_increase
            best_position = i

        # Early exit if cost increase is zero
        if min_increase == 0:
            break

    return best_position
