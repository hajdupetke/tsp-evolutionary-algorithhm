"""Tests for src/utils.py (Coder 1 — Phase 1)."""

from pathlib import Path
from src.utils import *

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_cities(tsp_instance):
    cities = tsp_instance["cities"]
    assert len(cities) == tsp_instance["n"]
    assert cities[0] == tsp_instance["first_city"]

def test_distance():
    assert distance((0, 0), (3, 4)) == 5.0
    assert distance((0, 0), (0, 0)) == 0.0

def test_calculate_cost(tsp_instance):
    cities = tsp_instance["cities"]
    n = tsp_instance["n"]
    tour = [i for i in range(n)]
    cost = calculate_cost(tour, cities)
    assert cost > 0
    assert isinstance(cost, float)

def test_calculate_fitness(tsp_instance):
    cities = tsp_instance["cities"]
    n = tsp_instance["n"]
    tour = [i for i in range(n)]
    cost = calculate_cost(tour, cities)
    fitness = calculate_fitness(tour, cities)
    assert fitness == 1.0 / cost