"""
utils.py — Data loading and tour evaluation utilities.

Provides:
  - load_cities: parse TSPLIB .tsp coordinate files (backward compatible)
  - load_instance: parse coordinates + EDGE_WEIGHT_TYPE
  - distance: raw Euclidean distance (kept for backward compatibility)
  - euc2d_rounded / geo_distance: TSPLIB-compliant edge weights
  - pairwise: dispatcher by weight type
  - build_distance_matrix: n x n TSPLIB edge-weight matrix
  - calculate_cost: total tour length (report Eq. 4, TSPLIB edge weights)
  - calculate_fitness: fitness = 1 / cost (report Eq. 5)

TSPLIB reference: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
Known optima (symmetric TSP list): berlin52 7542, kroA100 21282,
kroA200 29368 (EUC_2D), burma14 3323 (GEO).
"""

import math

RRR = 6378.388  # TSPLIB GEO earth radius


def load_cities(filename):
    """
    Read a TSPLIB .tsp file and return a list of (x, y) coordinates.

    Index in the returned list equals the city number (0-indexed).
    Lines before NODE_COORD_SECTION are skipped; reading stops at EOF.
    """
    cities, _ = load_instance(filename)
    return cities


def load_instance(filename):
    """
    Read a TSPLIB .tsp file and return (cities, weight_type).

    weight_type is 'EUC_2D' or 'GEO' (defaults to 'EUC_2D' if absent).
    Robust to 'KEY: VALUE' and 'KEY : VALUE' spellings.
    """
    cities = []
    weight_type = "EUC_2D"
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
                if "EDGE_WEIGHT_TYPE" in line and ":" in line:
                    weight_type = line.split(":", 1)[1].strip()
                continue
            if line.startswith("EOF"):
                break

            parts = line.split()
            # Format: city_id  x  y
            x = float(parts[1])
            y = float(parts[2])
            cities.append((x, y))

    return cities, weight_type


def distance(city1, city2):
    """
    Raw Euclidean distance between two cities given as (x, y) tuples.
    Kept for backward compatibility (e.g. tests).

    For TSPLIB-compliant tour costs use euc2d_rounded / pairwise instead.
    """
    return math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)


def euc2d_rounded(city1, city2):
    """TSPLIB EUC_2D edge weight: int(sqrt(dx^2+dy^2) + 0.5)."""
    return int(math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2) + 0.5)


def _geo_coord(x):
    """Convert TSPLIB GEO DDD.MM coordinate to radians."""
    deg = int(x)
    minutes = x - deg
    return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0


def geo_distance(city1, city2):
    """TSPLIB GEO edge weight (great-circle, RRR=6378.388)."""
    lat1 = _geo_coord(city1[0])
    lon1 = _geo_coord(city1[1])
    lat2 = _geo_coord(city2[0])
    lon2 = _geo_coord(city2[1])
    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    arg = max(-1.0, min(1.0, arg))  # clamp float noise
    return int(RRR * math.acos(arg) + 1.0)


def pairwise(city1, city2, weight_type="EUC_2D"):
    """Dispatch to the TSPLIB edge-weight function for weight_type."""
    if weight_type == "GEO":
        return geo_distance(city1, city2)
    return euc2d_rounded(city1, city2)


def build_distance_matrix(cities, weight_type="EUC_2D"):
    """Precompute the n x n TSPLIB edge-weight matrix (ints)."""
    n = len(cities)
    fn = geo_distance if weight_type == "GEO" else euc2d_rounded
    return [[0 if i == j else fn(cities[i], cities[j]) for j in range(n)] for i in range(n)]


def calculate_cost(tour, cities, weight_type="EUC_2D"):
    """
    Total tour distance — Equation (4) from the report, TSPLIB edge weights.

    C(pi) = sum of d(pi_k, pi_{k+1}) for k = 1..N-1  +  d(pi_N, pi_1)
    The last term closes the loop (return to the starting city).

    EUC_2D edges are int-rounded per TSPLIB, GEO uses great-circle;
    the total is returned as float (TSPLIB integer value, e.g. 7542.0).
    """
    if weight_type == "GEO":
        total = 0
        len_tour = len(tour)
        for k in range(len_tour - 1):
            total += geo_distance(cities[tour[k]], cities[tour[k + 1]])
        total += geo_distance(cities[tour[-1]], cities[tour[0]])
        return float(total)
    total = 0
    len_tour = len(tour)
    for k in range(len_tour - 1):
        total += euc2d_rounded(cities[tour[k]], cities[tour[k + 1]])
    # Close the loop: return from last city to first
    total += euc2d_rounded(cities[tour[-1]], cities[tour[0]])
    return float(total)


def calculate_fitness(tour, cities, weight_type="EUC_2D"):
    """
    Convert tour cost to fitness — Equation (5) from the report.

    F(pi) = 1 / C(pi)
    Shorter tours → smaller cost → higher fitness.
    """
    return 1.0 / calculate_cost(tour, cities, weight_type)