"""Tests for operators.py."""

import random

from src.operators import (
    tournament_selection,
    order_crossover,
    swap_mutation,
    inversion_mutation,
)


def is_valid(tour, n):
    return sorted(tour) == list(range(n))


def test_tournament_returns_index():
    rng = random.Random(1)
    costs = [100, 50, 75, 20, 90]

    for _ in range(100):
        i = tournament_selection(costs, 3, rng)
        assert isinstance(i, int)
        assert 0 <= i < len(costs)


def test_tournament_picks_lowest_cost():
    # Use fixed picks so we know which index should win.
    class FixedPicks:
        def __init__(self, picks):
            self.picks = list(picks)

        def randrange(self, n):
            return self.picks.pop(0)

    costs = [100, 50, 75, 20, 90]

    assert tournament_selection(costs, 3, FixedPicks([0, 2, 4])) == 2
    assert tournament_selection(costs, 2, FixedPicks([1, 3])) == 3


def test_tournament_prefers_low_cost():
    rng = random.Random(3)
    costs = [10, 1000, 1000, 1000, 1000]

    wins = sum(
        tournament_selection(costs, 3, rng) == 0
        for _ in range(1000)
    )

    assert wins > 400


def test_crossover_example():
    p1 = [0, 1, 2, 3, 4, 5, 6, 7]
    p2 = [3, 7, 5, 0, 6, 1, 2, 4]

    class FixedCuts:
        def __init__(self):
            self.calls = 0

        def randint(self, a, b):
            self.calls += 1
            return 2 if self.calls == 1 else 4

    child = order_crossover(p1, p2, FixedCuts())

    # Check both the permutation and the copied section.
    assert is_valid(child, 8)
    assert child[2:5] == [2, 3, 4]


def test_crossover_valid():
    rng = random.Random(4)
    n = 52

    for _ in range(1000):
        p1 = list(range(n))
        p2 = list(range(n))
        rng.shuffle(p1)
        rng.shuffle(p2)

        child = order_crossover(p1, p2, rng)
        assert is_valid(child, n)


def test_crossover_does_not_change_parents():
    rng = random.Random(5)
    p1 = list(range(20))
    p2 = list(range(20))
    rng.shuffle(p2)

    old_p1 = p1[:]
    old_p2 = p2[:]

    order_crossover(p1, p2, rng)

    assert p1 == old_p1
    assert p2 == old_p2


def test_swap_mutation():
    rng = random.Random(6)
    n = 52
    tour = list(range(n))

    for _ in range(1000):
        mutated = swap_mutation(tour, rng)

        assert is_valid(mutated, n)
        assert mutated is not tour
        assert tour == list(range(n))

        # A swap should change exactly two positions.
        differences = sum(a != b for a, b in zip(tour, mutated))
        assert differences == 2


def test_inversion_mutation():
    rng = random.Random(7)
    n = 52
    tour = list(range(n))

    for _ in range(1000):
        mutated = inversion_mutation(tour, rng)

        assert is_valid(mutated, n)
        assert mutated is not tour
        assert tour == list(range(n))


def test_same_seed_same_result():
    p1 = list(range(30))
    p2 = list(range(29, -1, -1))

    a = order_crossover(p1, p2, random.Random(42))
    b = order_crossover(p1, p2, random.Random(42))

    # Same seed should give the same crossover result.
    assert a == b