"""
utils.py — Data loading and tour evaluation utilities (Coder 1).

Provides:
  - load_cities: parse TSPLIB .tsp coordinate files
  - distance: Euclidean distance between two cities
  - calculate_cost: total tour length (report Eq. 4)
  - calculate_fitness: fitness = 1 / cost (report Eq. 5)
"""

import math


def load_cities(filename):
    """
    Read a TSPLIB .tsp file and return a list of (x, y) coordinates.

    Index in the returned list equals the city number (0-indexed).
    Lines before NODE_COORD_SECTION are skipped; reading stops at EOF.
    """
    cities = []
    in_section = False

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("NODE_COORD_SECTION"):
                in_section = True
                continue
            if not in_section:
                continue
            if line.startswith("EOF"):
                break

            parts = line.split()
            # Format: city_id  x  y
            x = float(parts[1])
            y = float(parts[2])
            cities.append((x, y))

    return cities

def distance(city1, city2):
    """
    Returns the Euclidean distance between two cities given as (x, y) tuples.
    Formula:
    Distance:
        d = sqrt((x₂ - x₁)² + (y₂ - y₁)²)

    :param city1: First city as an (x, y) tuple.
    :param city2: Second city as an (x, y) tuple.,
    """
    return math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)

def calculate_cost(tour, cities):
    """
    Total tour distance — Equation (4) from the report.

    C(pi) = sum of d(pi_k, pi_{k+1}) for k = 1..N-1  +  d(pi_N, pi_1)
    The last term closes the loop (return to the starting city).
    """
    total = 0.0
    len_tour = len(tour)
    for k in range(len_tour - 1):
        total += distance(cities[tour[k]], cities[tour[k + 1]])
    # Close the loop: return from last city to first
    total += distance(cities[tour[-1]], cities[tour[0]])
    return total

def calculate_fitness(tour, cities):
    """
    Convert tour cost to fitness — Equation (5) from the report.

    F(pi) = 1 / C(pi)
    Shorter tours → smaller cost → higher fitness.
    """
    return 1.0 / calculate_cost(tour, cities)