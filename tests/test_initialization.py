"""Tests for src/initialization.py"""

from src.utils import calculate_cost
from src.initialization import *


def test_random_population(tsp_instance):
    n = tsp_instance["n"]
    pop = random_population(20, n)
    assert len(pop) == 20
    assert len(pop[0]) == n
    assert all(len(set(tour)) == n for tour in pop)

def test_nearest_neighbour(tsp_instance):
    cities = tsp_instance["cities"]
    n = tsp_instance["n"]
    tour = nearest_neighbour_tour(cities, start_city=0)
    assert len(tour) == n
    assert len(set(tour)) == n
    # NN should beat a naive sequential tour
    nn_cost = calculate_cost(tour, cities)
    naive_cost = calculate_cost([i for i in range(n)], cities)
    assert nn_cost < naive_cost

def test_two_opt(tsp_instance):
    cities = tsp_instance["cities"]
    n = tsp_instance["n"]
    tour = nearest_neighbour_tour(cities, start_city=0)
    before = calculate_cost(tour, cities)
    # Cap iterations on larger instances so tests stay fast
    max_iter = 5 if n >= 100 else None
    improved = two_opt_improvement(tour, cities, max_iter=max_iter)
    after = calculate_cost(improved, cities)
    assert after <= before
    assert len(set(improved)) == n

def test_angle_based(tsp_instance):
    cities = tsp_instance["cities"]
    n = tsp_instance["n"]
    tour = angle_based_tour(cities)
    assert len(tour) == n
    assert len(set(tour)) == n

def test_smart_population(tsp_instance):
    cities = tsp_instance["cities"]
    n = tsp_instance["n"]
    # Keep population small — NN+2opt is expensive on kroA100/200
    pop_size = 4 if n >= 100 else 10
    pop = smart_population(cities, pop_size)
    assert len(pop) == pop_size
    assert all(len(set(tour)) == n for tour in pop)

    avg_smart = sum(calculate_cost(t, cities) for t in pop) / len(pop)
    pop_random = random_population(pop_size, n)
    avg_random = sum(calculate_cost(t, cities) for t in pop_random) / len(pop_random)
    assert avg_smart < avg_random