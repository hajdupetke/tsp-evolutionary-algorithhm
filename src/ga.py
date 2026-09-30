import time

def population_split(p, r):
    """
    Calculate how many individuals stay and how many children are created.
    """
    num_pairs = round(r * p / 2)
    num_children = 2 * num_pairs
    keep_n = p - num_children

    assert keep_n >= 0

    return keep_n, num_pairs


def random_search(D, n, evaluations, rng, cost_fn):
    """Generate random tours and keep the best one."""
    start = time.time()

    best_tour = None
    best_cost = float("inf")
    history_best = []
    history_gen = []

    for i in range(evaluations):
        tour = list(range(n))
        rng.shuffle(tour)

        cost = cost_fn(tour, D)

        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]

        history_best.append(best_cost)
        history_gen.append(best_cost)

    return {
        "best_tour": best_tour,
        "best_cost": best_cost,
        "history_best": history_best,
        "history_gen": history_gen,
        "seconds": time.time() - start,
    }