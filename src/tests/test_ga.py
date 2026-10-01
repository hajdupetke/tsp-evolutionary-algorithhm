"""Tests for src/ga.py."""

import math
import random

from src.ga import run_ga, random_search, population_split


def circle_distances(n, radius=1000):
    """Create distances for cities placed on a circle."""
    points = [
        (
            radius * math.cos(2 * math.pi * i / n),
            radius * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]

    return [
        [
            int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)
            for b in points
        ]
        for a in points
    ]


def tour_cost(tour, D):
    """Calculate the cost of a tour."""
    return sum(
        D[tour[i]][tour[(i + 1) % len(tour)]]
        for i in range(len(tour))
    )


def random_init(n, p, rng):
    """Create p random tours."""
    population = []

    for _ in range(p):
        tour = list(range(n))
        rng.shuffle(tour)
        population.append(tour)

    return population


PARAMS = {
    "p": 100,
    "r": 0.7,
    "m": 0.2,
    "G": 300,
    "k": 5,
    "mutation": "swap",
}


def test_population_split():
    for p in (50, 60, 100, 140, 150, 200):
        for r in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
            keep_n, num_pairs = population_split(p, r)

            assert keep_n + 2 * num_pairs == p
            assert keep_n >= 0
            assert num_pairs >= 0


def test_population_split_edges():
    assert population_split(100, 0.0) == (100, 0)

    keep_n, num_pairs = population_split(100, 1.0)
    assert keep_n == 0
    assert num_pairs == 50


def test_ga_finds_valid_tour():
    n = 20
    D = circle_distances(n)

    result = run_ga(
        D, PARAMS, random.Random(1), random_init, tour_cost
    )

    assert sorted(result["best_tour"]) == list(range(n))
    assert result["best_cost"] == tour_cost(result["best_tour"], D)


def test_ga_gets_close_to_optimum():
    n = 20
    D = circle_distances(n)
    optimum = tour_cost(list(range(n)), D)

    for seed in (1, 2, 3):
        result = run_ga(
            D, PARAMS, random.Random(seed), random_init, tour_cost
        )

        gap = (result["best_cost"] - optimum) / optimum * 100
        assert gap < 15.0


def test_ga_beats_random_search():
    n = 20
    D = circle_distances(n)
    evaluations = PARAMS["p"] * PARAMS["G"]

    ga = run_ga(
        D, PARAMS, random.Random(1), random_init, tour_cost
    )

    rs = random_search(
        D, n, evaluations, random.Random(1), tour_cost
    )

    assert ga["best_cost"] < rs["best_cost"]


def test_history_lists_have_correct_length():
    n = 20
    D = circle_distances(n)

    result = run_ga(
        D, PARAMS, random.Random(1), random_init, tour_cost
    )

    assert len(result["history_best"]) == PARAMS["G"]
    assert len(result["history_gen"]) == PARAMS["G"]

    history = result["history_best"]

    # The best cost should never get worse
    assert all(
        history[i + 1] <= history[i]
        for i in range(len(history) - 1)
    )

    assert history[-1] == result["best_cost"]


def test_population_size_stays_correct():
    n = 12
    D = circle_distances(n)

    params = dict(PARAMS, p=50, G=50)

    result = run_ga(
        D, params, random.Random(1), random_init, tour_cost
    )

    assert sorted(result["best_tour"]) == list(range(n))


def test_inversion_mutation_works():
    n = 20
    D = circle_distances(n)
    optimum = tour_cost(list(range(n)), D)

    params = dict(PARAMS, mutation="inversion")

    result = run_ga(
        D, params, random.Random(1), random_init, tour_cost
    )

    gap = (result["best_cost"] - optimum) / optimum * 100

    assert gap < 15.0


def test_same_seed_gives_same_result():
    n = 15
    D = circle_distances(n)
    params = dict(PARAMS, G=50)

    a = run_ga(
        D, params, random.Random(99), random_init, tour_cost
    )

    b = run_ga(
        D, params, random.Random(99), random_init, tour_cost
    )

    assert a["best_tour"] == b["best_tour"]
    assert a["history_gen"] == b["history_gen"]


def test_random_search_finds_valid_tour():
    n = 20
    D = circle_distances(n)

    result = random_search(
        D, n, 500, random.Random(1), tour_cost
    )

    assert sorted(result["best_tour"]) == list(range(n))
    assert result["best_cost"] == tour_cost(result["best_tour"], D)
