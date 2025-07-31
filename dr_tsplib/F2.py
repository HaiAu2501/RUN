# Final round optimized implementation for city_badness
# Strategy ID: F2
# Phase: Final round (system-aware)

import numpy as np

def city_badness(tour_idx: int, tour: list[int], distances: np.ndarray) -> float:
    n = len(tour)
    if n < 3:
        return float('inf')  # Cannot remove cities if less than 3 in tour

    city = tour[tour_idx]
    prev_city = tour[(tour_idx - 1) % n]
    next_city = tour[(tour_idx + 1) % n]

    # Compute savings from removing the city
    connection_savings = (distances[prev_city, city] + distances[city, next_city] - distances[prev_city, next_city])

    # Determine connectivity and distance metrics
    connectivity_degree = np.count_nonzero(distances[city])
    connected_distances = distances[city][distances[city] > 0]
    average_distance = np.mean(connected_distances) if connected_distances.size > 0 else 1  # Avoid division by zero
    distance_variance = np.var(connected_distances) if connected_distances.size > 0 else 0

    # Introduce a weighted score based on local visits
    local_visit_count = tour.count(city)
    local_density = local_visit_count / n

    # Adjusting penalties
    penalty_for_disconnection = (2.0 / (connectivity_degree + 1)) if connectivity_degree < 2 else 0
    frequent_visit_penalty = (np.log1p(local_visit_count) + (local_density * 3))  # Log scale for adapting to frequency

    # Final badness score calculation
    badness_score = (connection_savings + distance_variance)
    badness_score *= (1.0 + penalty_for_disconnection + frequent_visit_penalty)  # Combine various penalties

    return badness_score