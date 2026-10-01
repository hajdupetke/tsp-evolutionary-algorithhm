"""
initialization.py — Population initialization strategies.

Provides:
  - random_population: baseline random permutations
  - nearest_neighbour_tour: greedy NN heuristic (Liu, 2014)
  - two_opt_improvement: local search (Liu, 2014)
  - angle_based_tour: non-crossing geometric tour (Liao et al., 2012)
  - smart_population: half NN+2-opt, half angle-based
"""

import math
import random
from src.utils import calculate_cost, distance

DIVERSITY_SWAPS = 3


def random_population(population_size, num_cities, rng=None):
    """
    Baseline initialization — `population_size` random permutations
    of city indices 0 .. num_cities-1.

    :param rng: random.Random instance for reproducibility.
                If None, the global `random` module is used (backward
                compatible with the original signature).
    """
    rng = rng if rng is not None else random
    population = []
    for _ in range(population_size):
        tour = [i for i in range(num_cities)]
        rng.shuffle(tour)
        population.append(tour)
    return population

def nearest_neighbour_tour(cities, start_city=0):
    """
    Build one tour using the nearest-neighbour greedy heuristic (Liu, 2014).

    Starting from `start_city`, repeatedly append the closest unvisited city.
    """
    len_cities = len(cities)
    not_visited = set(range(len_cities))
    tour = [start_city]
    not_visited.remove(start_city)

    while not_visited:
        current = tour[-1]
        nearest = min(not_visited, key=lambda city: distance(cities[current], cities[city]))
        tour.append(nearest)
        not_visited.remove(nearest)

    return tour

def two_opt_improvement(tour, cities, max_iter=None):
    """
    2-opt local search (Liu, 2014).

    Repeatedly reverses tour segments [i..j] whenever doing so reduces cost.
    Restarts the search after every accepted improvement.
    """
    tour = tour[:]
    len_tour = len(tour)
    is_improved = True
    iters = 0

    while is_improved:
        if max_iter is not None and iters >= max_iter:
            break
        is_improved = False
        iters += 1

        for i in range(1, len_tour - 1):
            for j in range(i + 1, len_tour):
                # Reverse segment [i..j]
                new_tour = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                if calculate_cost(new_tour, cities) < calculate_cost(tour, cities):
                    tour = new_tour
                    is_improved = True
                    break  # restart after any improvement
            if is_improved:
                break

    return tour

def get_angle(idx, cities, cx, cy):
    """Polar angle of city `idx` relative to center (cx, cy), in degrees [0, 360)."""
    x, y = cities[idx]
    deg = math.degrees(math.atan2(y - cy, x - cx))
    return deg if deg >= 0 else deg + 360.0

def angle_based_tour(cities):
    """
    Build one non-crossing tour by sorting cities by polar angle
    from the geographic center (Liao et al., 2012, Section 3.1).
    """
    n = len(cities)
    cx = sum(c[0] for c in cities) / n
    cy = sum(c[1] for c in cities) / n

    tour = sorted(range(n), key=lambda idx: get_angle(idx, cities, cx, cy))
    return tour

def _swap_mutation_local(tour, rng=None):
    """Swap two random positions — used for diversity in smart_population."""
    rng = rng if rng is not None else random
    mutated = tour.copy()
    i, j = rng.sample(range(len(mutated)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def smart_population(cities, population_size, rng=None, max_iter=None):
    """
    Combined smart initialization (report Section 2.3).

    First half:  NN + 2-opt with different start cities.
    Second half: angle-based tour with a few random swaps for diversity.

    :param rng: random.Random instance for the diversity swaps.
                If None, the global `random` module is used.
    :param max_iter: forwarded to two_opt_improvement; caps local-search
                cost on large instances (e.g. kroA100/200).
    """
    len_cities = len(cities)
    half = population_size // 2
    population = []
    rng = rng if rng is not None else random

    # First half: nearest-neighbour + 2-opt
    for i in range(half):
        start_city = i % len_cities
        tour = nearest_neighbour_tour(cities, start_city)
        tour = two_opt_improvement(tour, cities, max_iter=max_iter)
        population.append(tour)

    # Second half: angle-based with light mutation for diversity
    base = angle_based_tour(cities)
    for _ in range(population_size - half):
        tour = base.copy()
        # A few random swaps so individuals are not identical
        for _ in range(DIVERSITY_SWAPS):
            tour = _swap_mutation_local(tour, rng)
        population.append(tour)

    return population