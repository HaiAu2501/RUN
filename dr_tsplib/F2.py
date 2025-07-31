# Final round optimized implementation for city_badness
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def city_badness(tour_idx: int, tour: list[int], distances: np.ndarray) -> float:
    n = len(tour)
    if n < 3:
        return float('inf')  # Handle edge case for insufficient cities

    city = tour[tour_idx]  # The evaluated city
    prev_city = tour[(tour_idx - 1) % n]  # Previous city
    next_city = tour[(tour_idx + 1) % n]  # Next city

    # Calculate local badness as the distance to direct neighbors
    local_badness = distances[prev_city, city] + distances[city, next_city]

    # Compute total distance to all cities
    total_distance = np.sum(distances[city])
    avg_distance = total_distance / (n - 1) if n > 1 else 0

    # Dynamic detour penalty based on neighboring connections
    detour_penalty = 0.0
    if local_badness > avg_distance:
        detour_penalty = 0.9 * (local_badness - avg_distance)

    # Connectivity penalty: focus on isolating cities more aggressively
    connectivity_penalty = 0.0
    if avg_distance > 0:
        connectivity_ratio = total_distance / (n * avg_distance)
        if connectivity_ratio > 1.2:  # Only increase for significant disconnections
            connectivity_penalty = 2.0 * connectivity_ratio

    # Adjust average penalty based on tour density 
    avg_penalty = 1.5 * (avg_distance / n)

    # Final badness score: combine metrics
    score = local_badness + avg_penalty + detour_penalty + connectivity_penalty
    return score